import pytest

from vpm2.timeline import plan_timeline, PlacedClip


def seg(i, start, end, duration):
    return {"id": i, "start": start, "end": end, "duration": duration}


def test_clip_fits_within_gap_plays_at_normal_speed():
    # segment 0..4 (gap to next at 5.0), clip is 3.0s -> fits
    segs = [seg(0, 0.0, 4.0, 3.0), seg(1, 5.0, 7.0, 1.0)]
    placed = plan_timeline(segs, video_duration=10.0)
    assert placed[0] == PlacedClip(id=0, start=0.0, speed=1.0,
                                   src_duration=3.0, out_duration=3.0)


def test_clip_uses_pause_before_next_segment():
    # clip 4.5s, next segment starts at 5.0 -> 5.0s available, still fits
    segs = [seg(0, 0.0, 4.0, 4.5), seg(1, 5.0, 7.0, 1.0)]
    placed = plan_timeline(segs, video_duration=10.0)
    assert placed[0].speed == 1.0
    assert placed[0].out_duration == 4.5


def test_overrun_accelerates_within_cap():
    # clip 6.0s, available 5.0s -> speed 1.2 (<=1.25), out 5.0s
    segs = [seg(0, 0.0, 4.0, 6.0), seg(1, 5.0, 7.0, 1.0)]
    placed = plan_timeline(segs, video_duration=10.0, max_speed=1.25)
    assert placed[0].speed == pytest.approx(1.2)
    assert placed[0].out_duration == pytest.approx(5.0)


def test_overrun_beyond_cap_clamps_and_pushes_next():
    # clip 10.0s, available 5.0s -> needs 2.0x but capped 1.25 -> out 8.0s
    segs = [seg(0, 0.0, 4.0, 10.0), seg(1, 5.0, 7.0, 1.0)]
    placed = plan_timeline(segs, video_duration=20.0,
                           max_speed=1.25, allow_push=True)
    assert placed[0].speed == pytest.approx(1.25)
    assert placed[0].out_duration == pytest.approx(8.0)
    # next clip pushed to start at 0 + 8.0 = 8.0 (after segment0 end on timeline)
    assert placed[1].start == pytest.approx(8.0)


def test_last_segment_bounded_by_video_duration():
    segs = [seg(0, 8.0, 9.0, 4.0)]
    placed = plan_timeline(segs, video_duration=10.0, max_speed=1.25)
    # available = 10.0 - 8.0 = 2.0, clip 4.0 -> needs 2.0x capped 1.25 -> out 3.2
    assert placed[0].start == pytest.approx(8.0)
    assert placed[0].speed == pytest.approx(1.25)
    assert placed[0].out_duration == pytest.approx(3.2)


def test_behind_schedule_clip_compresses_at_max_speed():
    # clip0 overruns hugely and pushes the cursor past clip1's whole slot, so
    # clip1 is "behind" (available <= 0). It must run at max_speed to claw time
    # back instead of drifting at 1.0x (the old bug).
    segs = [seg(0, 0.0, 4.0, 20.0), seg(1, 5.0, 7.0, 1.0), seg(2, 9.0, 11.0, 1.0)]
    placed = plan_timeline(segs, video_duration=30.0,
                           max_speed=1.25, allow_push=True)
    # clip0: available 5, src 20 -> 1.25, out 16, cursor 16
    # clip1: target max(5,16)=16, next start 9 -> available -7 -> behind
    assert placed[1].start == pytest.approx(16.0)
    assert placed[1].speed == pytest.approx(1.25)
    assert placed[1].out_duration == pytest.approx(0.8)


def test_no_push_keeps_next_at_original_start():
    segs = [seg(0, 0.0, 4.0, 10.0), seg(1, 5.0, 7.0, 1.0)]
    placed = plan_timeline(segs, video_duration=20.0,
                           max_speed=1.25, allow_push=False)
    assert placed[1].start == pytest.approx(5.0)