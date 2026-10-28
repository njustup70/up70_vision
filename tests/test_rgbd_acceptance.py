"""实际验收风险回归：录制不足、余量污染、漏复查与跨包标注覆盖。"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from lab.rgbd_check.annotate import _Window
from lab.rgbd_check.bag import FrameMeta, BagScan
from lab.rgbd_check.cli import main
from lab.rgbd_check.geometry import assess_geometry
from lab.rgbd_check.measure import edge_diagnostics
from lab.rgbd_check.pairing import pair_frames
from lab.rgbd_check.report import spatial_rows, refresh
from vision.depth.timing import capture_coverage


def frames(role, seconds):
    return [FrameMeta(role, i, 10**9 + round(t * 10**9), round(t * 10**9),
                      200, 200, 'bgr8' if role == 'color' else '16UC1', 600, 'cam')
            for i, t in enumerate(seconds)]


def test_perfect_pairs_do_not_hide_short_recording():
    result = pair_frames(frames('color', range(5)), frames('depth', range(5)),
                         warmup_ns=10**9, stable_duration_s=10)
    assert result.status == 'PASS'
    assert result.integrity['status'] == 'FAIL'
    assert result.integrity['streams']['color']['missing_end_s'] == 7


def test_tail_proves_coverage_without_entering_pair_denominator():
    result = pair_frames(frames('color', range(7)), frames('depth', range(7)),
                         warmup_ns=10**9, stable_duration_s=3)
    assert result.integrity['status'] == 'PASS'
    assert result.counts == {'color': 3, 'depth': 3}
    assert [p.color.index for p in result.pairs] == [1, 2, 3]
    assert result.tail_dropped == {'color': 3, 'depth': 3}
    assert result.warmup_dropped == {'color': 1, 'depth': 1}


def test_one_stream_stops_early():
    check = capture_coverage({'color': [0, 7*10**9], 'depth': [0, 4*10**9]}, 10**9, 5)
    assert check['status'] == 'FAIL'
    assert check['streams']['color']['missing_end_s'] == 0
    assert check['streams']['depth']['missing_end_s'] == 2


def test_unknown_duration_and_missing_stream_cannot_pass():
    assert capture_coverage({'color': [1], 'depth': [1]}, 0)['status'] == 'UNVERIFIED'
    assert capture_coverage({'color': [1, 10**10]}, 0, 1)['status'] == 'FAIL'


def test_integer_window_boundary_and_ten_second_trends():
    origin = 1793116073674378885
    offsets = list(range(0, 32*10**9, 100_000_000))
    color = frames('color', [t/10**9 for t in offsets])
    # 大整数只做整数平移，保留纳秒边界
    from dataclasses import replace
    color = [replace(f, bag_time_ns=f.bag_time_ns + origin) for f in color]
    depth = [replace(f, role='depth') for f in color]
    result = pair_frames(color, depth, warmup_ns=10**9, stable_duration_s=30)
    assert result.integrity['stable_end_bag_ns'] == origin + 31*10**9
    assert result.counts['color'] == 300
    freq = result.frequency['color']
    assert freq['header_fps'] == 10
    assert [w['frames'] for w in freq['windows_10s']] == [100, 100, 100]


def edge(edge_id='edge1', vertical=False, review='yes'):
    c = [[30, 20], [30, 150]] if vertical else [[20, 30], [150, 30]]
    d = [[32, 20], [32, 150]] if vertical else [[20, 32], [150, 32]]
    return {'edge_id': edge_id, 'color_segment': c, 'depth_segment': d,
            'confirmed': True, 'manual_review': review}


def report(root, session='a', condition='center_near', edges=None, geometry='PASS'):
    root.mkdir()
    (root/'selected').mkdir()
    entry = {'session': session, 'pair_id': 'c0_d0', 'condition_id': condition, 'image_size': [200, 200]}
    (root/'selected'/'c0_d0.json').write_text(json.dumps(entry))
    ann = dict(entry, edges=edges if edges is not None else [edge(), edge('edge2', True)])
    (root/'annotations.json').write_text(json.dumps({'frames': [ann]}))
    summary = {'analysis_version': 2, 'session': session, 'capture': {'mode': 'aligned'},
               'geometry': {'status': geometry}, 'sync': {'status': 'PASS'},
               'capture_integrity': {'status': 'PASS'}, 'selected': entry}
    (root/'summary.json').write_text(json.dumps(summary))
    refresh(root, 'aligned')
    return root


def test_no_manual_review_stays_unverified(tmp_path):
    out = report(tmp_path/'out', edges=[edge(review=None), edge('edge2', True)])
    rows = spatial_rows(out, 'aligned')
    assert rows[0]['measurement_status'] == 'PASS'
    assert rows[0]['status'] == 'UNVERIFIED'
    assert len(rows) == 6


def test_missing_second_edge_is_explicit(tmp_path):
    out = report(tmp_path/'out', edges=[edge()])
    rows = spatial_rows(out, 'aligned')
    assert rows[1]['edge_id'] == 'edge2'
    assert rows[1]['status'] == 'UNVERIFIED'
    assert json.loads((out/'summary.json').read_text())['spatial']['conditions']['center_near'] == 'UNVERIFIED'


@pytest.mark.parametrize('review', ['no', 'yes'])
def test_review_controls_condition_acceptance(tmp_path, review):
    out = report(tmp_path/'out', edges=[edge(review=review), edge('edge2', True, review)])
    result = json.loads((out/'summary.json').read_text())
    assert result['spatial']['conditions']['center_near'] == ('PASS' if review == 'yes' else 'FAIL')


def test_parallel_reference_edges_cannot_pass(tmp_path):
    out = report(tmp_path/'out', edges=[edge(), edge('edge2')])
    assert spatial_rows(out, 'aligned')[0]['status'] == 'UNVERIFIED'


def test_geometry_is_required_even_with_reviewed_small_offsets(tmp_path):
    out = report(tmp_path/'out', geometry='UNVERIFIED')
    assert spatial_rows(out, 'aligned')[0]['status'] == 'UNVERIFIED'


def test_wrong_session_annotation_cannot_be_applied(tmp_path):
    out = report(tmp_path/'out')
    a = json.loads((out/'annotations.json').read_text())
    a['frames'][0]['session'] = 'other'
    (out/'annotations.json').write_text(json.dumps(a))
    assert spatial_rows(out, 'aligned')[0]['offset_px'] is None


def test_summary_keeps_identical_pair_ids_from_three_sessions(tmp_path):
    paths = [report(tmp_path/c, session=c, condition=c) for c in ('center_near', 'center_far', 'right_mid')]
    out = tmp_path/'summary'
    assert main(['summarize', *map(str, paths), '--out', str(out)]) == 0
    s = json.loads((out/'summary.json').read_text())
    assert s['spatial']['status'] == 'PASS'
    assert len(s['spatial']['rows']) == 6
    assert len({(r['session'], r['pair_id']) for r in s['spatial']['rows']}) == 3
    assert main(['summarize', str(paths[0]), str(paths[0]), '--out', str(tmp_path/'bad')]) == 2
    assert not (tmp_path/'bad').exists()


def test_midpoint_close_can_hide_fanning():
    diag = edge_diagnostics([[0, 10], [100, 10]], [[0, 6], [100, 14]])
    assert abs(diag['angle_deg']) > 4
    assert diag['along_offsets_px'] == [-3.6, 0, 3.6]
    reverse = edge_diagnostics([[0, 10], [100, 10]], [[100, 14], [0, 6]])
    assert reverse == diag
    assert edge_diagnostics([[0, 10], [100, 10]], [[45, 12], [55, 12]])['along_offsets_px'] == [None, 2, None]


def test_window_review_and_unmeasurable_never_need_stdin(monkeypatch):
    monkeypatch.setattr('builtins.input', lambda *a: pytest.fail('窗口不应读终端'))
    w = _Window({'session': 'x', 'pair_id': 'c0_d0', 'condition_id': 'center_near'},
                np.zeros((200, 200, 3), np.uint8), np.ones((200, 200), np.uint16))
    w.state.add_edge('edge1', [[20,30],[150,30]], [[20,32],[150,32]])
    assert w.on_key(ord('y')) == 'save'
    assert w.state.edges[0]['manual_review'] == 'yes'
    w.on_key(ord('n'))
    assert w.state.edges[0]['manual_review'] == 'no'
    w.on_key(ord('u'))
    assert w.state.edges[1]['confirmed'] is False
    w.on_key(ord('r'))
    assert len(w.state.edges) == 1
    assert w.state.edges[0]['manual_review'] is None


def test_geometry_evidence_cannot_be_replaced_by_equal_K():
    f = frames('color', [0, 1])
    info = dict(width=200, height=200, frame_id='cam', d=[0.1], k=[100]*9)
    scan = BagScan(Path('.'), {}, frames={'color': f, 'depth': f},
                   infos={'color_info': info, 'depth_info': dict(info, d=[0])})
    meta = {'capture': {'mode': 'aligned'}}
    assert assess_geometry(scan, meta)['status'] == 'UNVERIFIED'
    meta['capture']['geometry_validation'] = dict(status='PASS', same_pixel_grid=True,
        target_pixel_grid='verified grid', depth_z_definition='target optical Z mm',
        backprojection='pinhole', optical_frame='cam', sources=['independent evidence'])
    assert assess_geometry(scan, meta)['status'] == 'PASS'
    scan.geometry_changes.append('CameraInfo changed')
    assert assess_geometry(scan, meta)['status'] == 'FAIL'


def test_bad_metadata_is_rejected_before_creating_report(tmp_path):
    bag=tmp_path/'bag';bag.mkdir()
    (bag/'meta.yaml').write_text('capture:\n  duration_s: .nan\n')
    assert main(['analyze',str(bag),'--out',str(tmp_path/'out')]) == 2
    assert not (tmp_path/'out').exists()


def test_scan_detects_later_camera_info_change(tmp_path, monkeypatch):
    from lab.rgbd_check import bag as module
    connections = [SimpleNamespace(topic='/'+r, msgtype=module.ROLE_TYPES[r], msgcount=2)
                   for r in module.ROLE_TYPES]
    def message(role, index):
        msg=SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=1,nanosec=index),frame_id='cam'),
                            width=200,height=200,encoding='bgr8',step=600)
        if role.endswith('info'):
            msg.distortion_model='plumb_bob';msg.k=[100.0+index]*9
            msg.d=[0.0]*5;msg.r=[0.0]*9;msg.p=[0.0]*12;msg.binning_x=msg.binning_y=0
        return msg
    class Reader:
        def __init__(self, paths):self.connections=connections
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def messages(self,connections):
            conn=connections[0]
            return [(conn,i,message(conn.topic[1:],i)) for i in range(2)]
        def deserialize(self,raw,msgtype):return raw
    monkeypatch.setattr(module,'_reader',lambda:Reader)
    result=module.scan(tmp_path,{r:'/'+r for r in module.ROLE_TYPES})
    assert any('color_info CameraInfo' in s for s in result.geometry_changes)
