def exists(val):
    return val is not None

Cache = namedtuple('Cache', ['cached_kvs', 'embeds'])

class BlockWithCacheAndEarlyExit(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.layers = nn.ModuleList([
            LayerNorm(config.n_embd, bias=config.bias),  # ln_1
            CausalSelfAttention(config),                 # attn
            LayerNorm(config.n_embd, bias=config.bias),  # ln_2
            MLP(config)                                  # mlp
        ])
        
        self.detach_early_exit_hiddens = config.detach_early_exit_hiddens
        self.early_exit_layer = config.early_exit_layer
        self.early_exit_transformer_blocks = ModuleList([])
        
        if self.early_exit_layer is not None:
            for _ in range(config.early_exit_extra_transformer_blocks):
                self.early_exit_transformer_blocks.append(ModuleList([
                    CausalAttention(dim=config.n_embd, dim_head=config.n_embd//config.n_head, heads=config.n_head),
                    FeedForward(dim=config.n_embd, mult=4)
                ]))
        
        self.cached_kvs = None

    def forward(self, x, start_from_early_exit_hiddens=False, early_exit_cache=None, cache=None, self_attn_kv_mask=None, return_early_exit_only=False):
        
        new_cached_kvs = []
        cache_kvs = cache_embeds = None

        if exists(cache):
            cache_kvs, cache_embeds = cache

        if exists(cache_kvs):
            iter_cache_kvs = iter(cache_kvs.unbind(dim = 1))
        else:
            iter_cache_kvs = iter([])
        
        layers = self.layers
        
        if start_from_early_exit_hiddens:
            assert not return_early_exit_only and exists(early_exit_cache)
            early_exit_layer_index = self.early_exit_layer

            early_cache_kvs, cache_embeds = early_exit_cache

            cache_embeds_len = cache_embeds.shape[-2]

            assert cache_embeds_len <= x.shape[-2]

            early_exit_layers, layers = layers[:early_exit_layer_index], layers[early_exit_layer_index:]
            x = x[:, cache_embeds_len:]

            if exists(early_cache_kvs):
                iter_early_cache_kvs = iter(early_cache_kvs.unbind(dim = 1))
            else:
                iter_early_cache_kvs = iter([])

            for ind, (attn, ff) in enumerate(early_exit_layers):
                residual = x
                attn_out, cached_kv = attn(x, context_mask = self_attn_kv_mask, cache = next(iter_early_cache_kvs, None))
                x = residual + attn_out

                new_cached_kvs.append(cached_kv)

                x = ff(x) + x

            x = torch.cat((cache_embeds, x), dim = -2)

        # if cache passed in, just use the last token, BLOCK

        if exists(cache):
            num_tokens_keep = x.shape[-2] - cache_kvs.shape[-2]
            x = x[:, -num_tokens_keep:]

        early_exit_hiddens = None

        # main transformer body
        # This will be replaced with: x, cache, early_exit = block(x, ...)
        # Goes into BLOCK

        for ind, (attn, ff) in enumerate(layers):
            layer = ind + 1

            residual = x
            attn_out, cached_kv = attn(x, cache = next(iter_cache_kvs, None))
            x = residual + attn_out

            new_cached_kvs.append(cached_kv)

            x = ff(x) + x

            if layer == self.early_exit_layer:
                early_exit_hiddens = x

                if self.detach_early_exit_hiddens:
                    early_exit_hiddens = early_exit_hiddens.detach()

                for early_exit_attn, early_exit_ff in self.early_exit_transformer_blocks:
                    residual = x
                    attn_out, cached_kv = early_exit_attn(x, rotary_emb = rotary_emb, cache = next(iter_cache_kvs, None))
                    x = residual + attn_out

                    new_cached_kvs.append(cached_kv)

                    x = early_exit_ff(x) + x

                if return_early_exit_only:
                    break

        return x, new_cached_kvs, early_exit_hiddens


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = False

    # Early exit parameters
    early_exit_layer = 4
    early_exit_extra_transformer_blocks =  0
    detach_early_exit_hiddens: bool = False

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.early_exit_layer is not None  # Make sure early_exit_layer is defined in config
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([BlockWithCacheAndEarlyExit(config) for _ in range(config.n_layer)]),  # Use BlockWithCacheAndEarlyExit
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight 

        if exists(config.early_exit_layer):
            self.to_early_exit_logits = nn.Sequential(
                    LayerNorm(config.n_embd, bias=False),
                    nn.Linear(config.n_embd, config.vocab_size, bias=False)
                )

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def forward(self, x, targets=False, return_loss=False, return_cache=False, seq_start_pos=None, cache=None, early_exit_cache=None, return_early_exit_only=False, start_from_early_exit_hiddens = False):

        device = x.device
        b, t = x.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer['wte'](x)
        pos_emb = self.transformer['wpe'](pos)
        
        x = self.transformer['drop'](tok_emb + pos_emb)

        self_attn_kv_mask = None
        if exists(seq_start_pos):
            batch, seq_len = x.shape[:2]
            seq_range = torch.arange(seq_len, device = x.device, dtype = torch.long)
            self_attn_kv_mask = seq_range >= seq_start_pos[..., None]
        
        full_cached_kvs = []

        for block in self.transformer.h:
            x, cached_kvs, early_exit_hiddens = block(x, start_from_early_exit_hiddens=start_from_early_exit_hiddens, early_exit_cache=early_exit_cache, cache=cache, self_attn_kv_mask=self_attn_kv_mask, return_early_exit_only=return_early_exit_only)

        x = self.transformer['ln_f'](x)

        full_cached_kvs = full_cached_kvs.append(cached_kvs)

        new_cached_kvs = torch.stack(full_cached_kvs, dim = 1)

        logits = self.lm_head(x)
        
        if not return_loss:
            if not return_cache:
                return logits

            if cache is not None:
                x = torch.cat((cache, x), dim=-2)  # Assuming the cache holds previous embeddings

            return logits, Cache(new_cached_kvs, x)

        # Compute the standard loss
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)  # Assuming ignore_index is -1

        # If there's no early exit logits conversion function, return only the standard loss
        if not hasattr(self, 'to_early_exit_logits'):
            return loss

        # Compute the early exit loss
        early_exit_logits = self.to_early_exit_logits(early_exit_hiddens)
        early_exit_loss = F.cross_entropy(early_exit_logits.view(-1, early_exit_logits.size(-1)), targets.view(-1), ignore_index=-1)  # Assuming ignore_index is -1

        return loss, early_exit_loss