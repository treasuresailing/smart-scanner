"""찰칵 문서 스캐너 — 사진 속 문서를 반듯하게 정리하고 이미지·PDF로 저장.

설치: python -m pip install --upgrade streamlit opencv-python-headless numpy pillow
실행: python -m streamlit run smart_scanner.py
자동 감지는 윤곽선·색상·배경 분리 기반이며 구김 펼치기·글자 방향 인식은 포함하지 않습니다.
"""
import hashlib
import io
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError


def ordered_points(points):
    """네 점을 중심 기준으로 정렬한 후 좌상단부터 시계방향으로 반환."""
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    return np.roll(pts, -int(np.argmin(pts.sum(axis=1))), axis=0).copy()


def valid_quad(points):
    pts = np.asarray(points, dtype=np.float32)
    return (pts.shape == (4, 2) and cv2.isContourConvex(pts)
            and abs(cv2.contourArea(pts)) > 100
            and np.min(np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1)) > 5)


@st.cache_data(show_spinner=False)
def detect_documents(image):
    """여러 이진화 결과에서 후보를 수집. 후보 순위는 인식 확률이 아님."""
    h, w = image.shape[:2]
    scale = min(1.0, 900.0 / max(h, w))
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    sh, sw = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks = [edges, binary, cv2.bitwise_not(binary)]
    # 밝기가 비슷해도 종이와 배경의 색상 차이가 남아 있을 수 있음.
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    for channel in (1, 2):
        if float(lab[:, :, channel].std()) < 2:
            continue
        _, color_mask = cv2.threshold(lab[:, :, channel], 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for color in (color_mask, cv2.bitwise_not(color_mask)):
            masks.append(cv2.morphologyEx(color, cv2.MORPH_CLOSE,
                                         np.ones((9, 9), np.uint8)))
    # 구겨진 종이의 경계가 끊겨도 색상·공간 정보를 함께 이용해 분리.
    # 종이 전체가 프레임 안에 있는 사진을 대상으로 함.
    if min(sh, sw) >= 50 and gray.std() > 3:
        gc_mask = np.zeros((sh, sw), np.uint8)
        margin = max(2, round(min(sh, sw) * 0.015))
        try:
            cv2.grabCut(small, gc_mask,
                        (margin, margin, sw - 2 * margin, sh - 2 * margin),
                        np.zeros((1, 65), np.float64),
                        np.zeros((1, 65), np.float64), 3, cv2.GC_INIT_WITH_RECT)
            foreground = np.uint8((gc_mask == cv2.GC_FGD)
                                  | (gc_mask == cv2.GC_PR_FGD)) * 255
            masks.append(cv2.morphologyEx(foreground, cv2.MORPH_CLOSE,
                                         np.ones((7, 7), np.uint8)))
        except cv2.error:
            pass
    candidates = []
    for mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:40]:
            if cv2.contourArea(contour) < sw * sh * 0.08:
                continue
            # 구김으로 윤곽점이 많아진 경우 볼록 껍질로 외곽 형태를 먼저 정리.
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)
            for epsilon in (0.015, 0.025, 0.04, 0.06):
                approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
                if len(approx) != 4:
                    continue
                pts = ordered_points(approx)
                if not valid_quad(pts):
                    continue
                # 사진의 바깥 테두리 자체를 문서로 선택하지 않도록 제외.
                boundary = ((pts[:, 0] <= 2) | (pts[:, 0] >= sw - 3)
                            | (pts[:, 1] <= 2) | (pts[:, 1] >= sh - 3))
                if np.all(boundary):
                    continue
                area = abs(cv2.contourArea(pts)) / (sw * sh)
                if area > 0.985:
                    continue
                lengths = np.linalg.norm(pts - np.roll(pts, 1, axis=0), axis=1)
                if lengths.min() / lengths.max() < 0.12:
                    continue
                # 영역 크기와 밝기를 후보 정렬의 단서로 사용.
                region = np.zeros_like(gray)
                cv2.fillConvexPoly(region, pts.astype(np.int32), 255)
                brightness = cv2.mean(gray, mask=region)[0] / 255.0
                # 배경의 일부가 이어져 생긴 부풀려진 사각형은 낮게 평가.
                coverage = min(1.0, abs(cv2.contourArea(contour)) /
                               max(abs(cv2.contourArea(pts)), 1))
                score = area * (0.8 + 0.2 * brightness) * coverage ** 2
                score *= 0.65 ** int(boundary.sum())
                original_pts = pts * np.array([w / sw, h / sh], np.float32)
                duplicate = next((i for i, (_, p) in enumerate(candidates)
                                  if np.mean(np.linalg.norm(original_pts - p, axis=1))
                                  < max(h, w) * 0.025), None)
                if duplicate is None:
                    candidates.append((score, original_pts))
                elif score > candidates[duplicate][0]:
                    candidates[duplicate] = (score, original_pts)
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [pts for _, pts in candidates[:5]]


def perspective(image, points, ratio):
    pts = np.asarray(points, dtype=np.float32)
    if not valid_quad(pts):
        raise ValueError('모서리 순서와 위치를 확인해 주세요.')
    tl, tr, br, bl = pts
    width = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    height = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if ratio == 'A4 세로':
        width = height / np.sqrt(2)
    elif ratio == 'A4 가로':
        height = width / np.sqrt(2)
    factor = min(1.0, 4000.0 / max(width, height))
    ow, oh = max(2, round(width * factor)), max(2, round(height * factor))
    dst = np.float32([[0, 0], [ow - 1, 0], [ow - 1, oh - 1], [0, oh - 1]])
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(image, matrix, (ow, oh), borderValue=(255, 255, 255))


def rotate_expand(image, angle):
    if angle == 0:
        return image.copy()
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    c, s = abs(matrix[0, 0]), abs(matrix[0, 1])
    nw, nh = int(np.ceil(w * c + h * s)), int(np.ceil(h * c + w * s))
    matrix[0, 2] += (nw - w) / 2
    matrix[1, 2] += (nh - h) / 2
    return cv2.warpAffine(image, matrix, (nw, nh), flags=cv2.INTER_CUBIC,
                          borderValue=(255, 255, 255))


def trim_edges(image, percent):
    """원근 보정 후 남은 가장자리 배경을 일정 비율만큼 자름."""
    h, w = image.shape[:2]
    dx, dy = round(w * percent / 100), round(h * percent / 100)
    return image[dy:h - dy, dx:w - dx].copy()


def soften_shadows(image, strength):
    """넓은 영역의 명도 편차 완화. 구김의 기하학적 복원은 하지 않음."""
    if strength <= 0:
        return image.copy()
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0].astype(np.float32)
    # 글자 크기의 어두운 성분을 줄인 후 넓은 조명 분포 추정.
    scale = min(1.0, 900 / max(light.shape))
    small = cv2.resize(light, None, fx=scale, fy=scale)
    k = max(5, int(min(small.shape) * 0.018) | 1)
    background = cv2.dilate(small, np.ones((k, k), np.uint8))
    background = cv2.GaussianBlur(background, (0, 0), max(3, min(small.shape) * 0.03))
    background = cv2.resize(background, (light.shape[1], light.shape[0]))
    corrected = np.clip(light * 245 / np.maximum(background, 30), 0, 255)
    lab[:, :, 0] = np.clip(light * (1 - strength) + corrected * strength, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def show_image(image, caption=None):
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=caption,
             width='stretch')


def change_rotation(degrees):
    """회전 버튼 클릭 횟수를 저장하여 다른 설정 변경 시에도 방향 유지."""
    st.session_state['edit_rotation'] = (st.session_state.get('edit_rotation', 0) + degrees) % 360


def reset_rotation():
    st.session_state['edit_rotation'] = 0
    st.session_state['edit_angle'] = 0.0


# 네 모서리 편집기는 이 파일 안에 포함되므로 추가 파일이 필요하지 않음.
CORNER_JS = r"""
export default function({parentElement, data, setStateValue}) {
  const canvas = parentElement.querySelector('canvas');
  const ctx = canvas.getContext('2d');
  const note = parentElement.querySelector('.note');
  const img = new Image();
  let pts = data.points.map(p => [...p]);
  let saved = pts.map(p => [...p]);
  let active = -1;
  function draw() {
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    ctx.beginPath();
    pts.forEach((p,i) => i ? ctx.lineTo(p[0]*canvas.width,p[1]*canvas.height)
                          : ctx.moveTo(p[0]*canvas.width,p[1]*canvas.height));
    ctx.closePath();ctx.fillStyle='rgba(0,160,120,.10)';ctx.fill();
    ctx.strokeStyle='#00b889';ctx.lineWidth=3;ctx.stroke();
    pts.forEach((p,i) => {
      const x=p[0]*canvas.width,y=p[1]*canvas.height;
      ctx.beginPath();ctx.arc(x,y,12,0,Math.PI*2);
      ctx.fillStyle='#fff';ctx.fill();ctx.strokeStyle='#007e63';ctx.stroke();
      ctx.fillStyle='#005b46';ctx.font='bold 13px sans-serif';
      ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(String(i+1),x,y);
    });
  }
  function valid() {
    const cross=pts.map((p,i)=>{
      const q=pts[(i+1)%4],r=pts[(i+2)%4];
      return (q[0]-p[0])*(r[1]-q[1])-(q[1]-p[1])*(r[0]-q[0]);
    });
    const area=Math.abs(pts.reduce((sum,p,i)=>{
      const q=pts[(i+1)%4];return sum+p[0]*q[1]-q[0]*p[1];
    },0))/2;
    return (cross.every(v=>v>0)||cross.every(v=>v<0)) && area>.002;
  }
  function position(e) {
    const b=canvas.getBoundingClientRect();
    return [Math.max(0,Math.min(1,(e.clientX-b.left)/b.width)),
            Math.max(0,Math.min(1,(e.clientY-b.top)/b.height))];
  }
  canvas.onpointerdown=e=>{
    const p=position(e),box=canvas.getBoundingClientRect();
    let distance=Infinity;
    pts.forEach((q,i)=>{
      const d=Math.hypot((p[0]-q[0])*box.width,(p[1]-q[1])*box.height);
      if(d<distance){distance=d;active=i;}
    });
    if(distance>40){active=-1;return;}
    saved=pts.map(p=>[...p]);canvas.setPointerCapture(e.pointerId);e.preventDefault();
  };
  canvas.onpointermove=e=>{if(active<0)return;pts[active]=position(e);draw();};
  canvas.onpointerup=e=>{
    if(active<0)return;
    active=-1;
    if(valid()){
      saved=pts.map(p=>[...p]); note.textContent='자르는 위치를 반영했습니다.';
      setStateValue('points',pts.map(p=>[...p]));
    }else{
      pts=saved.map(p=>[...p]);note.textContent='선이 교차하지 않도록 모서리를 옮겨 주세요.';
    }
    draw();
  };
  canvas.onpointercancel=()=>{active=-1;pts=saved.map(p=>[...p]);draw();};
  img.onload=()=>{canvas.width=img.naturalWidth;canvas.height=img.naturalHeight;draw();};
  img.src=data.image;
}
"""


@st.cache_resource
def corner_component():
    return st.components.v2.component(
        'smart_scanner_corners',
        html='<canvas aria-label="종이 모서리 조절"></canvas><div class="note">동그라미를 종이 모서리까지 끌어 놓으세요.</div>',
        css='canvas{display:block;width:100%;touch-action:none;cursor:crosshair;border-radius:8px;} .note{font:14px sans-serif;padding:10px 0;color:#555;}',
        js=CORNER_JS,
    )


def edit_corners(image, initial):
    """축소 화면 좌표를 정규화하여 원본 해상도의 좌표로 복원."""
    import base64
    h, w = image.shape[:2]
    epoch = st.session_state.get('_editor_epoch', 0)
    key = f'edit_canvas_{epoch}'
    normalized = (initial / np.array([w - 1, h - 1])).tolist()
    state = st.session_state.get(key, {})
    current = state.get('points', normalized)
    scale = min(1.0, 900 / max(h, w))
    display = cv2.resize(image, (round(w * scale), round(h * scale)))
    _, encoded = cv2.imencode('.jpg', display)
    result = corner_component()(
        data={'image': 'data:image/jpeg;base64,' + base64.b64encode(encoded).decode(),
              'points': current},
        default={'points': current}, key=key, on_points_change=lambda: None,
    )
    try:
        pts = np.asarray(result.points, np.float32) * np.array([w - 1, h - 1], np.float32)
        if valid_quad(pts) and np.isfinite(pts).all():
            return pts
    except (TypeError, ValueError):
        pass
    return initial


def create_pdf(image, paper='A4 자동 방향'):
    """보정 이미지를 한 페이지 PDF로 저장. 글자 검색용 OCR은 포함하지 않음."""
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if paper != '이미지에 맞춤':
        size = (1240, 1754) if rgb.height >= rgb.width else (1754, 1240)
        if paper == 'A4 세로':
            size = (1240, 1754)
        elif paper == 'A4 가로':
            size = (1754, 1240)
        page = Image.new('RGB', size, 'white')
        fitted = ImageOps.contain(rgb, size, Image.Resampling.LANCZOS)
        page.paste(fitted, ((size[0] - fitted.width) // 2, (size[1] - fitted.height) // 2))
        rgb = page
    output = io.BytesIO()
    rgb.save(output, format='PDF', resolution=150.0)
    return output.getvalue()


def reset_all():
    for key in list(st.session_state):
        if key.startswith('edit_'):
            del st.session_state[key]
    st.session_state['_editor_epoch'] = st.session_state.get('_editor_epoch', 0) + 1


def apply_design():
    """외부 이미지·폰트 없이 표시되는 가벼운 파스텔 인터페이스."""
    st.markdown('''
    <style>
    .stApp {background:#f7fbfa;color:#233c3a;}
    [data-testid="stSidebar"] {background:#edf6f2;border-right:1px solid #dcece5;}
    [data-testid="stSidebar"] h2 {color:#287364;}
    .block-container {max-width:1480px;padding-top:2rem;padding-bottom:3rem;}
    h1,h2,h3 {letter-spacing:-.035em;}
    h3 {color:#28695c;}
    .scanner-hero {display:flex;align-items:center;gap:24px;padding:30px 34px;
      margin-bottom:20px;border:1px solid #d4e9df;border-radius:26px;
      background:linear-gradient(115deg,#e4f5ed 0%,#f1faf5 66%,#fff0e4 100%);}
    .scanner-icon {position:relative;flex:0 0 64px;width:64px;height:82px;
      border:2px solid #64ac94;border-radius:12px;background:#fff;transform:rotate(-7deg);
      box-shadow:7px 7px 0 #cce8dc;}
    .scanner-icon:before {content:"";position:absolute;left:12px;right:12px;top:22px;
      height:4px;background:#a2cabb;box-shadow:0 12px 0 #bcdacd,0 24px 0 #bcdacd;border-radius:3px;}
    .scanner-icon:after {content:"✓";position:absolute;right:-12px;bottom:-7px;
      width:30px;height:30px;border-radius:50%;background:#ffcaad;color:#74432e;
      text-align:center;font:bold 22px sans-serif;transform:rotate(7deg);}
    .scanner-eyebrow {font-size:12px;font-weight:700;letter-spacing:.12em;color:#4d8271;margin-bottom:5px;}
    .scanner-name {font-size:clamp(26px,3.5vw,38px);font-weight:800;color:#225e50;line-height:1.3;}
    .scanner-description {font-size:15px;color:#527168;margin-top:8px;line-height:1.7;}
    .scanner-tags {display:flex;gap:8px;flex-wrap:wrap;margin-top:14px;}
    .scanner-tags span {background:#ffffffb8;border:1px solid #d5e9dd;border-radius:20px;
      padding:4px 11px;font-size:12px;color:#487261;}
    [data-testid="stFileUploader"] {background:#fff;border:1px dashed #a5cbbd;
      border-radius:18px;padding:16px;margin-bottom:16px;}
    [data-testid="stImage"] img {border-radius:16px;}
    [data-testid="stExpander"] {background:#fff;border-radius:14px;border-color:#dae8e0;}
    [data-testid="stButton"] button,[data-testid="stDownloadButton"] button {
      border-radius:12px;border-color:#c9dfd5;min-height:42px;color:#285f51;background:#fff;}
    [data-testid="stButton"] button:hover {border-color:#62ad91;background:#edf8f2;color:#1e604a;}
    [data-testid="stDownloadButton"] button {background:#ffdfc9;border-color:#f1c7a8;color:#70432c;font-weight:700;}
    [data-testid="stDownloadButton"] button:hover {background:#ffd0b0;border-color:#e8ad86;color:#59321f;}
    [data-baseweb="input"],[data-baseweb="select"] {border-radius:10px;}
    @media (max-width:600px) {
      .scanner-hero {padding:23px 19px;gap:18px;}
      .scanner-icon {flex-basis:45px;width:45px;height:65px;}
      .scanner-description {font-size:13px;}
    }
    @media (prefers-color-scheme:dark) {
      .stApp,[data-testid="stSidebar"] {color-scheme:light;}
    }
    </style>
    <div class="scanner-hero">
      <div class="scanner-icon" aria-hidden="true"></div>
      <div>
        <div class="scanner-eyebrow">PHOTO → SCAN</div>
        <div class="scanner-name" role="heading" aria-level="1">찰칵 문서 스캐너</div>
        <div class="scanner-description">사진 속 학습지와 문서를 반듯하게.<br>한 장의 사진을 이미지와 PDF로 간편하게 저장하세요.</div>
        <div class="scanner-tags"><span>종이 자동 찾기</span><span>회전 · 여백 정리</span><span>이미지 · PDF 저장</span></div>
      </div>
    </div>
    ''', unsafe_allow_html=True)


def main():
    st.set_page_config(page_title='찰칵 문서 스캐너', page_icon='📸', layout='wide')
    apply_design()
    upload = st.file_uploader('📷  1. 사진 올리기', type=['jpg', 'jpeg', 'png', 'webp'])
    if upload is None:
        st.info('학습지나 문서를 촬영한 사진을 올리면 종이 부분을 자동으로 찾아줍니다.')
        st.stop()
    raw = upload.getvalue()
    fingerprint = hashlib.sha256(raw).hexdigest()
    if st.session_state.get('_source_id') != fingerprint:
        reset_all()
        st.session_state['_source_id'] = fingerprint
    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > 40_000_000 or min(source.size) < 32:
                st.error('가로·세로 32px 이상, 4천만 화소 이하 이미지를 사용해 주세요.')
                st.stop()
            rgb = ImageOps.exif_transpose(source).convert('RGB')
            image = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error('사진을 읽지 못했습니다. JPG 또는 PNG 파일을 확인해 주세요.')
        st.stop()
    h, w = image.shape[:2]
    if max(h, w) > 4000:
        scale = 4000 / max(h, w)
        image = cv2.resize(image, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    h, w = image.shape[:2]
    with st.spinner('사진에서 종이를 찾고 있습니다…'):
        candidates = detect_documents(image)
    full = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    if 'edit_points' not in st.session_state:
        st.session_state['edit_points'] = (candidates[0] if candidates else full).tolist()
    with st.sidebar:
        st.header('🎨 사진 다듬기')
        st.button('처음 상태로 되돌리기', key='reset_all_button', on_click=reset_all)
        use_full = st.checkbox('자르지 않고 사진 전체 사용', key='edit_full')
        st.caption('구김 자체를 펴거나 글자 방향을 자동으로 판단하지는 않습니다.')
        mode = st.radio('사진 색상', ['컬러 유지', '회색조', '흑백 문서'], key='edit_color')
        soften = st.checkbox('종이의 어두운 그림자 줄이기', key='edit_soften')
        strength = st.slider('그림자 줄이는 정도', 0, 100, 60, 5,
                             disabled=not soften, key='edit_shadow_strength')
        st.caption('강도를 높이면 연한 글씨나 그림도 달라질 수 있습니다.')
        with st.expander('크기와 가장자리 조절'):
            ratio_label = st.selectbox('종이 모양', ['자동으로 맞추기', 'A4 세로', 'A4 가로'], key='edit_ratio')
            st.caption('A4를 선택하면 사진을 해당 가로·세로 비율에 맞춥니다.')
            resize_enabled = st.checkbox('저장 크기 직접 조절', key='edit_resize_enabled')
            target_width = st.number_input('저장 이미지의 가로 크기 (px)', 200, 4000, 1200, 100,
                                           disabled=not resize_enabled, key='edit_width')
            st.caption('기본은 보정 후 크기를 유지합니다. 크기를 줄이면 작은 글씨가 덜 선명해질 수 있습니다.')
            edge_trim = st.slider('가장자리를 조금 더 자르기 (%)', 0.0, 5.0, 0.0, 0.2, key='edit_edge_trim')
            st.caption('종이 끝도 함께 잘립니다. 배경이 조금 남았을 때만 높이세요.')
    st.subheader('✨  2. 반듯하게 다듬기')
    if not candidates and not use_full:
        st.warning('종이를 자동으로 찾지 못했습니다. 아래 ‘자르는 위치 수정하기’에서 모서리를 직접 맞춰 주세요.')
    manual = st.checkbox('자르는 위치 수정하기', key='edit_manual', disabled=use_full)
    points = np.asarray(st.session_state['edit_points'], np.float32)
    if manual and not use_full:
        st.caption('사진 위 동그라미 4개를 종이 모서리로 끌어 놓으면 결과가 바뀝니다.')
        if hasattr(st.components, 'v2'):
            points = edit_corners(image, points)
            st.session_state['edit_points'] = points.tolist()
        else:
            st.error('마우스로 모서리를 조절하려면 Streamlit 업데이트가 필요합니다: python -m pip install --upgrade streamlit')
    ratio = '문서 모서리 기준 추정' if ratio_label == '자동으로 맞추기' else ratio_label
    result = image.copy() if use_full else perspective(image, points, ratio)
    if edge_trim > 0:
        result = trim_edges(result, edge_trim)
    r1, r2, r3, r4 = st.columns(4)
    r1.button('↶ 왼쪽으로 돌리기', key='edit_rotate_left', on_click=change_rotation, args=(-90,))
    r2.button('↷ 오른쪽으로 돌리기', key='edit_rotate_right', on_click=change_rotation, args=(90,))
    r3.button('위아래 뒤집기 (180°)', key='edit_rotate_half', on_click=change_rotation, args=(180,))
    r4.button('회전 취소', key='edit_rotate_reset', on_click=reset_rotation)
    with st.expander('기울기와 여백 더 조절하기'):
        if 'edit_angle' not in st.session_state:
            st.session_state['edit_angle'] = 0.0
        angle = st.slider('기울기 맞추기 (°)', -15.0, 15.0, step=0.1, key='edit_angle')
        st.caption('음수는 오른쪽, 양수는 왼쪽으로 조금씩 기울입니다.')
        left = st.slider('왼쪽 여백 자르기 (%)', 0, 40, 0, key='edit_left')
        right = st.slider('오른쪽 여백 자르기 (%)', 0, 40, 0, key='edit_right')
        top = st.slider('위쪽 여백 자르기 (%)', 0, 40, 0, key='edit_top')
        bottom = st.slider('아래쪽 여백 자르기 (%)', 0, 40, 0, key='edit_bottom')
    result = np.ascontiguousarray(np.rot90(result, k=-(st.session_state.get('edit_rotation', 0) // 90)))
    result = rotate_expand(result, angle)
    rh, rw = result.shape[:2]
    result = result[round(rh * top / 100):round(rh * (1 - bottom / 100)),
                    round(rw * left / 100):round(rw * (1 - right / 100))]
    rh, rw = result.shape[:2]
    if resize_enabled:
        target_height = max(1, round(rh * target_width / rw))
        if target_height > 6000:
            st.error('세로가 너무 긴 이미지입니다. 저장 가로 크기를 줄여 주세요.')
            st.stop()
        if target_width != rw:
            result = cv2.resize(result, (int(target_width), target_height),
                                interpolation=cv2.INTER_AREA if target_width < rw else cv2.INTER_CUBIC)
    if soften:
        result = soften_shadows(result, strength / 100)
    if mode != '컬러 유지':
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        if mode == '흑백 문서':
            gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 31, 12)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    preview = image.copy()
    if not use_full:
        cv2.polylines(preview, [points.astype(np.int32)], True, (0, 180, 0), max(2, max(h,w)//400))
    a, b = st.columns(2)
    with a:
        st.markdown('**📷 올린 사진**')
        show_image(preview, '초록 선 안쪽을 남깁니다.' if not use_full else '사진 전체를 사용합니다.')
    with b:
        st.markdown('**✨ 완성할 문서**')
        show_image(result, f'{result.shape[1]} × {result.shape[0]}px')
    st.subheader('📥  3. 완성한 문서 저장하기')
    name = st.text_input('저장할 파일 이름', value=Path(upload.name).stem + '_scan', key='edit_filename')
    name = name.strip().removesuffix('.pdf').removesuffix('.png').removesuffix('.jpg')
    for character in '/\\:*?"<>|':
        name = name.replace(character, '_')
    name = name or 'scanned_document'
    image_type = st.radio('이미지 파일 형식', ['PNG', 'JPG'], horizontal=True, key='edit_format')
    paper = st.selectbox('PDF 용지', ['A4 자동 방향', 'A4 세로', 'A4 가로', '이미지에 맞춤'], key='edit_paper')
    st.caption('PDF는 현재 사진 한 장으로 만듭니다. A4는 비율을 유지해 용지 안에 배치하므로 흰 여백이 생길 수 있습니다.')
    ext = '.png' if image_type == 'PNG' else '.jpg'
    ok, encoded = cv2.imencode(ext, result)
    p1, p2 = st.columns(2)
    if ok:
        p1.download_button(f'{image_type} 이미지 저장', encoded.tobytes(), name + ext,
                           'image/png' if ext == '.png' else 'image/jpeg', key='download_image')
    pdf = create_pdf(result, paper)
    p2.download_button('PDF 문서 저장', pdf, name + '.pdf', 'application/pdf', key='download_pdf')


if __name__ == '__main__':
    main()
