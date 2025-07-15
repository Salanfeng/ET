import torch


def smart_mask_updater(
    ar_len, atten_mask, pos, k_caches, v_caches, new_k_caches, new_v_caches
):
    # Update the KV cache input for the next inference when the position exceeds the autoregressive length.
    if pos >= ar_len:
        for i, k_cache in enumerate(k_caches):
            k_cache[:, :, pos - ar_len] = new_k_caches[i][:, :, 0]

        for i, v_cache in enumerate(v_caches):
            v_cache[:, pos - ar_len, :] = new_v_caches[i][:, 0, :]
        atten_mask[:, :, pos - ar_len] = 0

    pos += 1
    return (atten_mask, pos, k_caches, v_caches)


def shift_pointer_updater(
    ar_len, atten_mask, pos, k_caches, v_caches, new_k_caches, new_v_caches
):
    # Update the KV cache input for the next inference when the position exceeds the autoregressive length.
    if pos >= ar_len:
        k_caches = [
            torch.cat([k_cache[:, :, 1:], new_k_caches[i][:, :, :1]], dim=-1)
            for i, k_cache in enumerate(k_caches)
        ]
        v_caches = [
            torch.cat([v_cache[:, 1:, :], new_v_caches[i][:, :1, :]], dim=1)
            for i, v_cache in enumerate(v_caches)
        ]
        atten_mask[:, :, -pos - 1] = 0

    pos += 1
    return (atten_mask, pos, k_caches, v_caches)



def main():
    # 假设有2层，每层4个head，cache长度为5，head_dim为3
    batch = 1
    num_heads = 4
    cache_len = 5
    head_dim = 3

    k_caches = [
        torch.full((batch, num_heads, cache_len), fill_value=10 + i)
        for i in range(2)
    ]
    # v_caches: [batch, cache_len, head_dim]，每层初始值不同
    v_caches = [
        torch.full((batch, cache_len, head_dim), fill_value=100 + i * 10)
        for i in range(2)
    ]

    # new_k_caches: [batch, num_heads, 1]，每层不同
    new_k_caches = [
        torch.full((batch, num_heads, 1), fill_value=1000 + i * 100)
        for i in range(2)
    ]
    # new_v_caches: [batch, 1, head_dim]，每层不同
    new_v_caches = [
        torch.full((batch, 1, head_dim), fill_value=2000 + i * 100)
        for i in range(2)
    ]

    # atten_mask: [batch, num_heads, cache_len]
    atten_mask = torch.ones(batch, num_heads, cache_len)

    ar_len = 3
    pos = 3  # 让 pos >= ar_len，触发更新

    print("Before update:")
    print("k_caches[0]:\n", k_caches[0])
    print("v_caches[0]:\n", v_caches[0])
    print("atten_mask:\n", atten_mask)

    atten_mask, pos, k_caches, v_caches = smart_mask_updater(
        ar_len, atten_mask, pos, k_caches, v_caches, new_k_caches, new_v_caches
    )

    print("\nAfter update:")
    print("k_caches[0]:\n", k_caches[0])
    print("v_caches[0]:\n", v_caches[0])
    print("atten_mask:\n", atten_mask)
    print("pos:", pos)

if __name__ == "__main__":
    main()