def exists(val):
    return val is not None


class Decoder(Module):

    #Init part, this is with the GPT config
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        ignore_index = -1,
        early_exit_layer = None,
        early_exit_extra_transformer_blocks = 0,
        detach_early_exit_hiddens = False
    ):
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        self.layers = ModuleList([])

        self.rotary_emb = RotaryEmbedding(dim = dim_head)

        for _ in range(depth):
            self.layers.append(ModuleList([
                CausalAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

        #Unsure where this would go

        self.detach_early_exit_hiddens = detach_early_exit_hiddens
        self.early_exit_layer = early_exit_layer
        self.to_early_exit_logits = None
        self.early_exit_transformer_blocks = ModuleList([])

        if exists(early_exit_layer):
            for _ in range(early_exit_extra_transformer_blocks):
                self.early_exit_transformer_blocks.append(ModuleList([
                    CausalAttention(dim = dim, dim_head = dim_head, heads = heads, rotary_emb = rotary_emb),
                    FeedForward(dim = dim, mult = ff_mult)
                ]))

            self.to_early_exit_logits = nn.Sequential(
                RMSNorm(dim),
                nn.Linear(dim, num_tokens, bias = False)
            )

        self.ignore_index = ignore_index

    def forward(
        self,
        x,
        return_loss = False,
        return_cache = False,
        seq_start_pos = None,
        cache = None,
        early_exit_cache = None,
        return_early_exit_only = False,
        start_from_early_exit_hiddens = False
    ):
        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        # handle seq start pos offset, goes in GPT module

        self_attn_kv_mask = None
        if exists(seq_start_pos):
            batch, seq_len = x.shape[:2]
            seq_range = torch.arange(seq_len, device = x.device, dtype = torch.long)
            self_attn_kv_mask = seq_range >= seq_start_pos[..., None]

        # relative positional encoding, goes nowhere for now

        rotary_emb = self.rotary_emb(x.shape[-2])

        # setup cache, goes in both, partly

        new_cached_kvs = []

        cache_kvs = cache_embeds = None

        if exists(cache):
            cache_kvs, cache_embeds = cache

        if exists(cache_kvs):
            iter_cache_kvs = iter(cache_kvs.unbind(dim = 1))
        else:
            iter_cache_kvs = iter([])

        # handle if previous cached embedding layer from early exit layer passed in
        #Goes into BLOCK

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
                attn_out, cached_kv = attn(x, context_mask = self_attn_kv_mask, rotary_emb = rotary_emb, cache = next(iter_early_cache_kvs, None))
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
            attn_out, cached_kv = attn(x, rotary_emb = rotary_emb, cache = next(iter_cache_kvs, None))
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

        #THIS ALL GOES INTO GPT MODULE 

        new_cached_kvs = torch.stack(new_cached_kvs, dim = 1)

        to_logits = self.to_logits if not return_early_exit_only else self.to_early_exit_logits

        logits = to_logits(x)

        if not return_loss:
            if not return_cache:
                return logits

            if exists(cache_embeds):
                x = torch.cat((cache_embeds, x), dim = -2)

            return logits, Cache(new_cached_kvs, x)

        loss = F.cross_entropy(
            rearrange(logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        if not exists(self.to_early_exit_logits):
            return loss

        early_exit_logits = self.to_early_exit_logits(early_exit_hiddens)

        early_exit_loss = F.cross_entropy(
            rearrange(early_exit_logits, 'b n c -> b c n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss, early_exit_loss