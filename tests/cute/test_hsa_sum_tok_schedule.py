import torch


def test_hsa_sum_tok_schedule_matches_dense_reference():
    from flash_attn.cute.hsa import (
        build_hsa_schedule_from_sum_tok_metadata,
        compute_hsa_sum_tok_mask,
        infer_hsa_sum_tok_metadata,
        schedule_to_attend_mask,
    )

    ht_id = 99
    eos_id = 2
    input_batch = torch.tensor(
        [
            [10, 11, ht_id, 12, 13, ht_id, ht_id, 14, ht_id, eos_id],
            [20, ht_id, 21, 22, ht_id, ht_id, 23, 24, ht_id, eos_id],
        ],
        dtype=torch.long,
    )
    token_level, seg_ids = infer_hsa_sum_tok_metadata(
        input_batch,
        eos_id=eos_id,
        ht_id=ht_id,
    )

    expected = compute_hsa_sum_tok_mask(token_level, seg_ids)
    schedule = build_hsa_schedule_from_sum_tok_metadata(token_level, seg_ids)
    actual = schedule_to_attend_mask(schedule)

    torch.testing.assert_close(actual, expected)


def test_hsa_sum_tok_schedule_noncausal_matches_dense_reference():
    from flash_attn.cute.hsa import (
        build_hsa_sum_tok_schedule,
        compute_hsa_sum_tok_mask,
        infer_hsa_sum_tok_metadata,
        schedule_to_attend_mask,
    )

    ht_id = 99
    input_batch = torch.tensor([[10, 11, ht_id, 12, 13, ht_id, ht_id]])
    token_level, seg_ids = infer_hsa_sum_tok_metadata(input_batch, ht_id=ht_id)

    expected = compute_hsa_sum_tok_mask(token_level, seg_ids, causal=False)
    schedule = build_hsa_sum_tok_schedule(input_batch, ht_id=ht_id, causal=False)
    actual = schedule_to_attend_mask(schedule)

    torch.testing.assert_close(actual, expected)
