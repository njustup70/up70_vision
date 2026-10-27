"""将 analyze 保存的数据绘成 PNG 和可离线查看的 HTML 报告。"""
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_rows(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def number(row, key):
    return float(row[key]) if row[key] else np.nan


def geometry(depth, labels, data):
    """重建 ROI 点云；顶层按 core 的桌面距离百分位定义，不冒充薄片。"""
    info, params = data['camera_info'], data['params']
    h, w = depth.shape
    rh, rw = int(h * params['roi_ratio']), int(w * params['roi_ratio'])
    y0, x0 = (h-rh)//2, (w-rw)//2
    roi = depth[y0:y0+rh, x0:x0+rw]
    v, u = np.nonzero(roi)
    z = roi[v, u].astype(float)
    k = info['k']
    pts = np.stack(((u+x0-k[2])*z/k[0], (v+y0-k[5])*z/k[4], z), axis=1)
    chosen = data['median_depth_target']
    is_box = np.zeros(len(z), dtype=bool)
    is_top = np.zeros(len(z), dtype=bool)
    if chosen is not None:
        is_box = labels[v, u] == chosen['label']
        plane = data['median_depth_plane']
        dist = np.abs(pts[is_box] @ np.asarray(plane['normal']) + plane['d'])
        if dist.size:
            cutoff = np.percentile(dist, 100 * (1 - params['top_ratio']))
            is_top[is_box] = dist >= cutoff
    return pts, is_box, is_top, (u+x0, v+y0), (x0, y0, rw, rh)


def overview(out, depth, data, points, is_top, pixels, roi):
    """原始深度、真实选层与候选位置。"""
    valid = depth > 0
    low, high = np.percentile(depth[valid], [2, 98])
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    im = ax.imshow(np.ma.masked_where(~valid, depth), cmap='turbo_r', vmin=low, vmax=high)
    fig.colorbar(im, ax=ax, label='Depth Z (mm)')
    x0,y0,w,h=roi
    ax.add_patch(plt.Rectangle((x0,y0), w,h, fill=False, color='white', lw=1.2))
    u,v=pixels
    ax.scatter(u[is_top][::3], v[is_top][::3], s=1, color='magenta', label='Top layer (20%)')
    info=data['camera_info']['k']
    def project(p):
        return info[0]*p[0]/p[2]+info[2], info[4]*p[1]/p[2]+info[5]
    chosen=data['median_depth_target']
    for c in data['median_depth_clusters']:
        p=np.asarray(c['top_center_mm']); uv=project(p)
        ax.scatter(*uv,s=55,facecolors='none',edgecolors='yellow')
        ax.annotate(f"{c['label']} {c['score']:.2f}",uv,color='white',fontsize=7)
    if chosen:
        p=np.asarray(chosen['top_center_mm']); n=np.asarray(chosen['top_normal']); start=project(p); end=project(p+60*n)
        ax.scatter(*start,s=100,marker='+',color='lime',label='Selected center')
        ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'lime','lw':2})
    ax.set(title='Depth median (first 60 frames): top layer and candidates',xlabel='u (px)',ylabel='v (px)')
    ax.legend(loc='lower left'); fig.savefig(out/'overview.png',dpi=160); plt.close(fig)
    fig, ax=plt.subplots(figsize=(8,6),constrained_layout=True)
    ax.imshow(np.ma.masked_where(~valid,depth),cmap='turbo_r',vmin=low,vmax=high)
    ax.scatter(u[is_top][::2],v[is_top][::2],s=2,color='magenta')
    if chosen:
        p=np.asarray(chosen['top_center_mm']); start=project(p); end=project(p+60*np.asarray(chosen['top_normal']))
        ax.scatter(*start,s=160,marker='+',color='lime'); ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'lime','lw':2})
        mask = (data['_label_map'] == chosen['label'])
        yy,xx=np.nonzero(mask)
        if len(xx):
            margin=35; ax.set_xlim(max(0,x0+xx.min()-margin),min(depth.shape[1],x0+xx.max()+margin)); ax.set_ylim(min(depth.shape[0],y0+yy.max()+margin),max(0,y0+yy.min()-margin))
    ax.set(title='Top layer detail (magenta): selection, not truth',xlabel='u (px)',ylabel='v (px)')
    fig.savefig(out/'top_detail.png',dpi=160); plt.close(fig)


def pointcloud(out, pts, box, top, data):
    fig=plt.figure(figsize=(14,7),constrained_layout=True)
    chosen=data['median_depth_target']
    # 远处背景在 4 米外，会把 0.5 米的盒子挤成一个点；显示范围明确截到 1.2 米。
    view = (pts[:, 2] <= 1200) & (np.abs(pts[:, 0]) <= 600) & (np.abs(pts[:, 1]) <= 600)
    for i,(elev,azim) in enumerate([(25,-65),(10,30)],1):
        ax=fig.add_subplot(1,2,i,projection='3d')
        # 其余点包含桌面与背景；按平面距离明确区分，避免把背景叫桌面。
        plane=data['median_depth_plane']; table=np.abs(pts @ np.asarray(plane['normal'])+plane['d']) < data['params']['plane_thresh_mm']
        for mask,color,label in [(view & table & ~box,'#8d99ae','Table'),(view & ~table & ~box,'#dddddd','Other'),(view & box & ~top,'#dd9b41','Object'),(view & top,'#d900ae','Top layer')]:
            p=pts[mask][::max(1,int(mask.sum()/4500))]
            if len(p): ax.scatter(p[:,0],p[:,1],p[:,2],s=1,c=color,alpha=.6,label=label)
        ax.scatter([0],[0],[0],c='black',s=40,marker='x',label='Optical origin')
        if chosen:
            p=np.asarray(chosen['top_center_mm']); n=np.asarray(chosen['top_normal'])
            ax.scatter(*p,color='limegreen',s=50,label='Center'); ax.quiver(*p,*n,length=100,color='green',linewidth=2)
            ax.text(*p,f"  ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f})",fontsize=8)
        ax.set(xlabel='X right (mm)',ylabel='Y down (mm)',zlabel='Z forward (mm)',title=f'Optical frame - view {i}, Z <= 1200 mm',
               xlim=(-600,600), ylim=(-600,600), zlim=(0,1200))
        ax.set_box_aspect((1,1,1))
        ax.view_init(elev=elev,azim=azim); ax.legend(fontsize=7,loc='upper left')
    fig.savefig(out/'pointcloud.png',dpi=150); plt.close(fig)


def stability(out, rows, data):
    fig,axs=plt.subplots(5,1,figsize=(13,13),sharex=True,constrained_layout=True)
    t=np.array([(int(r['timestamp_ns'])-int(rows[0]['timestamp_ns']))/1e9 for r in rows])
    median=data['single_frame_summary']['position_median_mm']
    for axis,key in enumerate(('x_mm','y_mm','z_mm')):
        values=np.array([number(r,key) for r in rows]); axs[0].plot(t,values,label=key[0].upper(),lw=.8)
        if median: axs[1].plot(t,values-median[axis],label=key[0].upper(),lw=.8)
    normals=np.array([[number(r,k) for k in ('nx','ny','nz')] for r in rows])
    valid=np.isfinite(normals).all(axis=1)
    if valid.any():
        n=np.median(normals[valid],axis=0); n=n/np.linalg.norm(n)
        unit=normals/np.linalg.norm(normals,axis=1)[:,None]
        axs[2].plot(t,np.degrees(np.arccos(np.clip(unit @ n,-1,1))),color='purple')
    axs[3].plot(t,[number(r,'score') for r in rows],label='selected score')
    axs[3].step(t,[int(r['positive_candidates']) for r in rows],label='score > 0',where='mid',alpha=.65)
    axs[3].step(t,[int(r['qualified_candidates']) for r in rows],label='score >= 0.5',where='mid',alpha=.65)
    status_values={'ok':0,'missing':1,'ambiguous':2,'failed':3,'invalid_normal':4}
    axs[4].scatter(t,[status_values[r['status']] for r in rows],s=7,c=[('green' if r['status']=='ok' else 'red') for r in rows])
    axs[4].set_yticks(list(status_values.values()),list(status_values.keys()))
    for ax,y in zip(axs,['Position (mm)','Deviation (mm)','Normal change (deg)','Score / count','Status']):
        ax.set_ylabel(y); ax.grid(alpha=.2)
    axs[0].legend(); axs[1].legend(); axs[3].legend(); axs[-1].set_xlabel('Header time from first frame (s)')
    fig.suptitle('Single-frame results; blanks are failed or ambiguous frames')
    fig.savefig(out/'stability.png',dpi=150); plt.close(fig)


def checks_plot(out,data):
    fig,ax=plt.subplots(figsize=(13,3.5),constrained_layout=True); ax.axis('off')
    names=['Height','Camera tilt','Top dimensions','Top / table angle','Candidates']
    def clean(s):
        s = str(s).replace('×', ' x ').replace('°', ' deg').replace('（', ' (').replace('）', ')')
        return re.sub(r'[^\x20-\x7e]', '', s).strip() or '--'
    table=[[names[i],clean(r['measured']),clean(r['expected']),clean(r['error']),r['status']] for i,r in enumerate(data['checks'])]
    if data['truth']['height_mm'] is not None:
        table[0][2] = f"{data['truth']['height_mm']:.1f} mm (+/- 2 mm)"
    if data['truth']['tilt_deg'] is not None:
        table[1][2] = f"{data['truth']['tilt_deg']:.1f} deg (+/- 3 deg)"
    table.append(['Absolute XYZ','--','No ground truth','--','UNVERIFIED'])
    tab=ax.table(cellText=table,colLabels=['Check','Measured','Reference / tolerance','Error','Status'],loc='center',cellLoc='center',colWidths=[.2,.18,.34,.15,.13]); tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.8)
    ax.set_title('Independent ruler checks - depth median (60 frames)'); fig.savefig(out/'checks.png',dpi=150);plt.close(fig)


def sample_plot(out, arrays):
    fig,axs=plt.subplots(1,3,figsize=(15,4),constrained_layout=True)
    for ax,key in zip(axs,['sample_first','sample_middle','sample_last']):
        a=arrays[key]; valid=a>0; lo,hi=np.percentile(a[valid],[2,98]); ax.imshow(np.ma.masked_where(~valid,a),cmap='turbo_r',vmin=lo,vmax=hi); ax.set_title(key); ax.axis('off')
    fig.suptitle('Visual stationarity check: per-image depth scale');fig.savefig(out/'samples.png',dpi=140);plt.close(fig)


def make_html(out,data):
    summary=data['single_frame_summary']; escape=lambda x:html.escape(str(x))
    bag_name = escape(Path(data['bag']).name)
    scene_note = ('本 bag 元数据记录为无目标物块：合格候选属于疑似误检，位置和法向不能视为物块位姿。'
                  if data['truth']['n_targets'] == 0 else '物块尺寸、姿态及目标数量以元数据为准；缺少参考值时不作精度判定。')
    if bag_name == 'raw_001':
        scene_note += '首、中、末帧画面变化明显，拍摄过程需复核，不能把全段统计当作稳定空场景误检率。'
    checks=''.join(f"<tr><td>{escape(c['check'])}</td><td>{escape(c['measured'])}</td><td>{escape(c['expected'])}</td><td>{escape(c['error'])}</td><td>{escape(c['status'])}</td></tr>" for c in data['checks'])
    figs=[('overview.png','60 帧深度中位数与顶层'),('top_detail.png','顶面选点局部'),('pointcloud.png','深度光学系点云'),('stability.png','单帧位置与候选稳定性'),('checks.png','历史尺测检验'),('samples.png','首、中、末帧静止检查')]
    images=''.join(f'<h2>{escape(label)}</h2><img src="{name}" alt="{escape(label)}">' for name,label in figs)
    text=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{bag_name} 深度位姿报告</title><style>body{{font:16px sans-serif;max-width:1150px;margin:auto;padding:2em;color:#162338;line-height:1.6}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:.5em}}code,pre{{white-space:pre-wrap}}.notice{{background:#fff2d5;padding:1em}}</style><h1>{bag_name} 深度位姿验证</h1><p>输入：{escape(data['bag'])}；提交：{escape(data['revision'])}；工作区有未提交修改：{escape(data['working_tree_dirty'])}。算法参数、内参及逐帧数据见 <a href="summary.json">summary.json</a> 和 <a href="frames.csv">frames.csv</a>。</p><p class="notice">{escape(scene_note)} 独立 XYZ 真值缺失：绝对位置精度及 M2 未验证。单帧抖动是重复性，不是准确度；相机姿态及场景静止需人工复核。深度流畸变和校正关系尚未核实。图中紫色依据保存的桌面平面近似重建最高 20% 选点，不代表人工标注真值。</p><h2>逐帧统计</h2><pre>{escape(json.dumps(summary,ensure_ascii=False,indent=2))}</pre><p>状态：ok = 唯一合格候选；missing = 无合格候选；ambiguous = 多候选；invalid_normal = 法向未朝相机。相邻有效帧跳变超过 30mm 仅标记为疑似切换，不自动删除。</p><h2>60 帧深度中位数检查</h2><table><tr><th>项目</th><th>实测</th><th>参考</th><th>误差</th><th>状态</th></tr>{checks}</table><p>候选数沿用历史 score &gt; 0 的规则；逐帧输出使用 ≥0.5，两个口径各自报告。如果提供了倾角参考值，机身与光轴的夹角偏差及测量不确定度仍需核查。</p>{images}<h2>下一步</h2><p>固定相机，补采有尺测网格与垫高的多位置 bag，每段记录尺寸、位置、测量不确定度及相同配置；据此计算相对位移误差。要验证绝对 XYZ，另需建立独立光学坐标基准。</p></html>'''
    (out/'report.html').write_text(text,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory',type=Path); args=p.parse_args(); out=args.directory
    data=json.loads((out/'summary.json').read_text(encoding='utf-8')); rows=load_rows(out/'frames.csv'); arrays=np.load(out/'visual_data.npz')
    depth=arrays['median_depth']; labels=arrays['label_map']; data['_label_map']=labels
    pts,box,top,pixels,roi=geometry(depth,labels,data)
    overview(out,depth,data,pts,top,pixels,roi);pointcloud(out,pts,box,top,data);stability(out,rows,data);checks_plot(out,data);sample_plot(out,arrays)
    del data['_label_map'];make_html(out,data);print(f'Report: {out / "report.html"}')


if __name__=='__main__':
    main()
