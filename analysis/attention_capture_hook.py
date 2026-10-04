# Diagnostic overlay appended to the pinned attention module in a Linux image.
# It captures only the fourth layer at prefix120, then calls the original kernel.
import os as _capture_os
from pathlib import Path as _CapturePath

_capture_original_attention = extend_attention_fwd_unified  # noqa: F821
_capture_matching_calls = 0


def extend_attention_fwd_unified(*args, **kwargs):
    global _capture_matching_calls
    destination = _capture_os.environ.get("SUFFIX_ATTENTION_CAPTURE_FILE")
    selected = False
    if destination and int(args[9].item()) == 120:
        _capture_matching_calls += 1
        selected = _capture_matching_calls == 4
    if selected:
        assert not _CapturePath(destination).exists()
        q, output, keys, values = args[:4]
        length = int(args[7][-1].item())
        indices = args[8][:length].long()
        capture = dict(
            q=q.detach().cpu(), q_stride=list(q.stride()),
            keys=keys[indices].detach().cpu(), values=values[indices].detach().cpu(),
            kv_indices=indices.detach().cpu(), prefix=int(args[9].item()),
            width=q.shape[0], layer_ordinal=3,
            custom_mask=kwargs.get("custom_mask"), sm_scale=kwargs["sm_scale"],
        )
        if capture["custom_mask"] is not None:
            capture["custom_mask"] = capture["custom_mask"].detach().cpu()
    result = _capture_original_attention(*args, **kwargs)
    if selected:
        capture["output"] = output.detach().cpu()
        torch.save(capture, destination)  # noqa: F821
    return result
