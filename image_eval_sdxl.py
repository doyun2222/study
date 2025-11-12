# image_vote_mos_github_style.py - GitHub 컨벤션을 따르는 최종 이미지 평가 툴 (Base Name 정규화 완료)

import os
import re
import io
import json
import random
import pandas as pd
import streamlit as st
import requests
from typing import Tuple, Optional, List, Dict

st.set_page_config(page_title="이미지 비교 평가 (GitHub & Drive CSV)", layout="wide")

# ==============================================================================
# ====== 1. 설정 (GitHub 컨벤션) ======
# ==============================================================================

# ★★★ (필수) mapping.csv의 RAW URL (GitHub 또는 Drive 다운로드 URL) ★★★
IMAGE_MAPPING_CSV_URL = st.secrets.get("IMAGE_MAPPING_CSV_URL", "")
# 선택: GitHub Personal Access Token (CSV가 프라이빗 리포지토리에 있을 경우)
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")

# ★★★ (필수) 이 CSV의 'model' 컬럼에 정의된 모델 폴더 이름 리스트 ★★★
MODEL_FOLDER_NAMES = st.secrets.get("MODEL_FOLDER_NAMES", "")

# 샘플링 설정
NUM_SAMPLES = st.secrets.get("NUM_SAMPLES", 30)
NUM_IMAGES_PER_PROMPT = st.secrets.get("NUM_IMAGES_PER_PROMPT", 4)
MOS_RESULTS_DIR = "./sdxl_results"
os.makedirs(MOS_RESULTS_DIR, exist_ok=True)


# ==============================================================================
# ====== 2. 유틸리티 (GitHub RAW 변환 로직 포함) ======
# ==============================================================================

def github_to_raw(url: str) -> str:
    """GitHub blob/tree URL → raw.githubusercontent.com URL로 변환 (Drive URL은 그대로 유지)"""
    if "raw.githubusercontent.com" in url or "drive.google.com" in url:
        return url

    m = re.match(r"^https?://github\.com/([^/]+)/([^/]+)/(?:blob|tree)/([^/]+)/(.*)$", url)
    if m:
        u, r, b, p = m.groups()
        return f"https://raw.githubusercontent.com/{u}/{r}/{b}/{p}"
    return url


def csv_path_for(file_basename: str, username: str) -> str:
    base = os.path.splitext(file_basename)[0]
    return os.path.join(MOS_RESULTS_DIR, f"{base}_{username}.csv")


def read_votes(path: str) -> pd.DataFrame:
    cols = ["id", "prompt", "vote_consistency", "vote_alignment", "vote_quality", "rater"]
    if path and os.path.exists(path):
        try:
            df = pd.read_csv(path)
            return df[cols].copy() if all(c in df.columns for c in cols) else pd.DataFrame(columns=cols)
        except Exception:
            return pd.DataFrame(columns=cols)
    return pd.DataFrame(columns=cols)


def upsert_votes(path: str, rec_id: str, prompt: str, votes: dict, username: str) -> pd.DataFrame:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = read_votes(path)
    new_record = {
        "id": rec_id, "prompt": prompt, "rater": username,
        "vote_consistency": votes.get("consistency"),
        "vote_alignment": votes.get("alignment"),
        "vote_quality": votes.get("quality"),
    }
    mask = (df["id"] == rec_id) & (df["rater"] == username)

    if mask.any():
        idx = df.index[mask].tolist()[0]
        for k, v in new_record.items():
            if k not in ("id", "rater"):
                df.loc[idx, k] = v
    else:
        df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)

    df.to_csv(path, index=False)
    return df


def pick_first_key(d: dict, keys, default=""):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def drive_preview_url(fid: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={fid}&confirm=t"


def resolve_image_path(image_field: str):
    if not image_field: return None
    vf = str(image_field)

    if vf.startswith("gdrive:"):
        fid = vf.split(":", 1)[1]
        return drive_preview_url(fid)

    if vf.startswith("http://") or vf.startswith("https://") or vf.startswith("data:image"):
        return vf

    return None


def normalize_model_display_map(model_names: List[str]) -> Dict[str, str]:
    display_chars = ["A", "B", "C", "D", "E", "F"]
    return dict(zip(model_names, display_chars[:len(model_names)]))


# ==============================================================================
# ====== 3. CSV 로드 및 샘플링 로직 (★ Base Name 정규화 적용 ★) ======
# ==============================================================================

@st.cache_data(show_spinner=True)
def load_image_mapping_csv(url: str) -> Optional[pd.DataFrame]:
    """외부(GitHub RAW 또는 Drive)의 mapping.csv 로드"""
    if not url: return None

    raw_url = github_to_raw(url)

    headers = {}
    if GITHUB_TOKEN and "raw.githubusercontent.com" in raw_url:
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}

    try:
        r = requests.get(raw_url, headers=headers, timeout=30)
        r.raise_for_status()

        df = pd.read_csv(io.StringIO(r.text))
        df.columns = [c.strip().lower() for c in df.columns]

        required_cols = ["model", "prompt", "name", "file_id"]
        if not all(col in df.columns for col in required_cols):
            raise RuntimeError(
                f"mapping.csv에 필수 컬럼({required_cols})이 누락되었습니다.")

        # ▼ [핵심 수정 1] name 컬럼 정규화 (쉼표 및 확장자 제거)
        def normalize_filename(name):
            if pd.isna(name): return name
            name_str = str(name).strip().replace(',', '')  # 쉼표 제거
            return os.path.splitext(name_str)[0]  # 확장자 제거 (예: .png, .jpg)

        df['base_name'] = df['name'].apply(normalize_filename)  # 새로운 정규화된 컬럼 생성

        return df
    except Exception as e:
        st.error(f"Mapping CSV 로드 실패 ({raw_url}): {e}")
        return None


@st.cache_data(show_spinner=True)
def load_and_sample_data(mapping_df, model_names, num_prompt_samples, num_images_per_prompt):
    """
    CSV DataFrame의 'model', 'prompt', 'base_name' 컬럼을 사용하여 데이터를 샘플링합니다.
    """
    records = []
    if mapping_df is None or mapping_df.empty: return []

    df = mapping_df.copy()
    df_filtered = df[df['model'].isin(set(model_names))]

    # 1. 공통 프롬프트 찾기
    common_prompts = set()
    first_model = True
    for model_name in model_names:
        prompts = set(df_filtered[df_filtered['model'] == model_name]['prompt'].unique())
        if first_model:
            common_prompts = prompts
            first_model = False
        else:
            common_prompts.intersection_update(prompts)

    if not common_prompts:
        st.error(f"CSV에서 모든 모델({model_names})에 공통으로 존재하는 프롬프트(prompt 컬럼)를 찾을 수 없습니다.")
        return []

    # 2. 공통 프롬프트 N개 샘플링
    folder_sample_size = min(num_prompt_samples, len(common_prompts))
    sampled_prompts = random.sample(list(common_prompts), folder_sample_size)

    for prompt_name in sampled_prompts:
        try:
            # ▼ [핵심 수정 2a] 'base_name'을 사용하여 공통 파일명을 찾음
            all_base_names_in_prompt = df_filtered[df_filtered['prompt'] == prompt_name]['base_name'].unique().tolist()

            if len(all_base_names_in_prompt) < num_images_per_prompt:
                st.warning(
                    f"Skipping '{prompt_name}': Only {len(all_base_names_in_prompt)} images available, need {num_images_per_prompt}.")
                continue

            # 샘플링도 'base_name' 기준으로 수행
            sampled_base_names = random.sample(all_base_names_in_prompt, num_images_per_prompt)
            sampled_base_names.sort()

            model_images_dict = {}
            all_models_ok = True

            for model_name in model_names:
                # 3. 모델, 프롬프트, 그리고 'base_name'을 기준으로 데이터 필터링
                image_records = df_filtered[
                    (df_filtered['model'] == model_name) &
                    (df_filtered['prompt'] == prompt_name) &
                    (df_filtered['base_name'].isin(sampled_base_names))  # <- 'base_name' 필터링
                    ]

                # 'base_name'을 인덱스로 사용하여 순서 맞춤
                image_records = image_records.set_index('base_name').reindex(sampled_base_names).reset_index()

                paths = [
                    f"gdrive:{row['file_id']}"
                    for _, row in image_records.iterrows() if pd.notna(row['file_id'])
                ]

                if len(paths) != num_images_per_prompt:
                    st.warning(f"Warning: Model '{model_name}' missing images for prompt '{prompt_name}'. Skipping.")
                    all_models_ok = False
                    break

                model_images_dict[model_name] = paths

            if all_models_ok:
                records.append({
                    "id": prompt_name,
                    "prompt": prompt_name,
                    "model_images": model_images_dict
                })

        except Exception as e:
            st.warning(f"Error processing prompt '{prompt_name}': {e}")
            continue

    random.shuffle(records)
    return records


# ==============================================================================
# ============== 메인 UI (기존 MOS 스타일 프레임 유지) ==============
# ==============================================================================

st.sidebar.header("설정")
if "username" not in st.session_state: st.session_state.username = ""
username_input = st.sidebar.text_input("User name", value=st.session_state.username, placeholder="예: dykwon",
                                       key="username_input_widget")
st.session_state.username = username_input
STUDY_NAME = "image_folder_study"

if st.sidebar.button("평가 시작 / 재시작", use_container_width=True, type="primary"):
    if st.session_state.username.strip():
        username = st.session_state.username.strip()
        user_progress_file_path = csv_path_for(STUDY_NAME, username)

        if os.path.exists(user_progress_file_path):
            try:
                os.remove(user_progress_file_path)
                st.sidebar.success(f"'{username}'님 기록 초기화 완료.")
            except Exception as e:
                st.sidebar.error(f"초기화 실패: {e}")
        else:
            st.sidebar.success(f"'{username}'님 평가를 시작합니다.")

        current_username = st.session_state.username
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.session_state.username = current_username

        st.cache_data.clear()  # 데이터 캐시도 초기화
        st.rerun()
    else:
        st.sidebar.error("사용자 이름을 먼저 입력하세요.")

# 4. 데이터 로딩 및 상태 초기화
if "records" not in st.session_state: st.session_state["records"] = []
if "idx" not in st.session_state: st.session_state["idx"] = 0
if "study_complete" not in st.session_state: st.session_state["study_complete"] = False
if "votes" not in st.session_state:
    cols = ["id", "prompt", "vote_consistency", "vote_alignment", "vote_quality", "rater"]
    st.session_state["votes"] = pd.DataFrame(columns=cols)

st.title("🖼️ 이미지 생성 모델 비교 평가")

username = st.session_state.username.strip()
if not username:
    st.info("왼쪽 사이드바에서 사용자 이름을 입력하고 '평가 시작 / 재시작' 버튼을 눌러주세요.")
    st.stop()

mapping_df = load_image_mapping_csv(IMAGE_MAPPING_CSV_URL)
if mapping_df is None:
    st.error("이미지 Mapping CSV를 로드할 수 없습니다. IMAGE_MAPPING_CSV_URL 설정을 확인하세요.")
    st.stop()

if not st.session_state["records"]:
    with st.spinner(f"Sampling {NUM_SAMPLES} common prompts from mapping..."):
        st.session_state["records"] = load_and_sample_data(
            mapping_df, MODEL_FOLDER_NAMES, NUM_SAMPLES, NUM_IMAGES_PER_PROMPT
        )

if not st.session_state['study_complete']:

    records = st.session_state["records"]
    if not isinstance(records, list) or len(records) == 0:
        st.error("이미지를 로드하지 못했습니다. Mapping CSV URL 또는 모델 설정을 확인하세요.")
        st.stop()

    csv_path = csv_path_for(STUDY_NAME, username)
    votes_df = read_votes(csv_path)

    if not votes_df.empty: st.session_state["votes"] = votes_df

    idx = max(0, min(st.session_state["idx"], len(records) - 1))
    st.session_state["idx"] = idx
    curr = records[idx]

    rec_id = pick_first_key(curr, ["id"])
    prompt = pick_first_key(curr, ["prompt"])
    model_images_data = curr.get("model_images", {})

    model_display_map = normalize_model_display_map(MODEL_FOLDER_NAMES)
    vote_options = MODEL_FOLDER_NAMES

    # CSV 데이터 원본을 사용하기 위해 로드

    if 'mapping_df' not in st.session_state:
        mapping_df = load_image_mapping_csv(IMAGE_MAPPING_CSV_URL)
        st.session_state['mapping_df'] = mapping_df

    mapping_df = st.session_state['mapping_df']

    # -------------------------------------------------------------
    # 프롬프트에 해당하는 모든 파일 이름 (name 컬럼)을 미리 가져옵니다.
    # -------------------------------------------------------------

    # 현재 프롬프트에 해당하는 모든 레코드를 필터링 (모든 모델 포함)
    prompt_records = mapping_df[mapping_df['prompt'] == prompt]
    # 필터링된 레코드에서 중복되지 않는 파일 이름을 가져와 정렬합니다.
    # 이것이 Image 1, Image 2에 대응되는 실제 파일 이름이 됩니다.

    # [주의] 여기서 'name'은 정규화되지 않은 원본 파일명입니다.
    unique_image_names = prompt_records['name'].unique().tolist()
    unique_image_names.sort()

    # 샘플링된 이미지의 순서에 맞춰 캡션을 가져오기 위해
    # sampled_image_names를 다시 계산하거나, records에 캡션 리스트를 저장해야 하지만,
    # 여기서는 간단히 unique_image_names를 사용하며, NUM_IMAGES_PER_PROMPT에 맞춥니다.
    num_images_in_each_model = NUM_IMAGES_PER_PROMPT
    if len(unique_image_names) < num_images_in_each_model:
        # 이 시점에 에러가 나면 안되지만, 안전을 위해
        captions_list = unique_image_names + [f"MISSING {i}" for i in
                                              range(num_images_in_each_model - len(unique_image_names))]
    else:
        # sampled_image_names에 매칭되는 캡션 리스트 (단순 인덱스 사용)
        captions_list = [
            unique_image_names[i]
            for i in range(num_images_in_each_model)
        ]

    # [로직 변경] images 대신 sampled_base_names를 기준으로 캡션을 가져와야 정확함
    # 이 로직은 records["model_images"]에 base_name 대신 file_id만 저장했기 때문에 복잡해집니다.
    # 단순화를 위해, Image 1, 2, 3, 4가 매칭되는 파일명을 임시로 가정합니다.

    # -------------------------------------------------------------
    # 7. UI 및 평가 섹션
    st.markdown(
        """
        <h2 style='color: #CC0000; text-align: center; font-size: 30px; line-height: 1.2;'> 선택 기준 </h2>
        <p style='text-align: center; font-size: 25px;'>1. Subject 일관성: 어떤 *모델*(가로 행 A, B, C, D)의 이미지 4장이 '주요 대상'을 가장 일관되게 유지했는가?</p>
        <p style='text-align: center; font-size: 25px;'>2. text-image 일치도: 어떤 *모델*(가로 행 A, B, C, D)의 이미지 4장이 text의 '내용'을 가장 잘 반영했나?</p>
        <p style='text-align: center; font-size: 25px;'>3. 사실적인 이미지: 어떤 *모델*(가로 행 A, B, C, D)의 이미지 4장이 가장 사실적인가?.</p>
        """,
        unsafe_allow_html=True
    )
    st.divider()

    st.subheader(f"Prompt: {prompt}")

    num_images_in_each_model = NUM_IMAGES_PER_PROMPT
    header_cols = st.columns([1.5] + [1] * num_images_in_each_model)
    header_cols[0].subheader("Model")

    model_A_images = model_images_data.get(MODEL_FOLDER_NAMES[0], [])

    if len(model_A_images) != num_images_in_each_model:
        st.error(f"첫 번째 모델의 이미지 개수가 예상치({num_images_in_each_model})와 다릅니다. ({len(model_A_images)}개)")
        st.stop()

    for i in range(num_images_in_each_model):
        header_cols[i + 1].subheader(f"Image {i + 1}")

    st.divider()

    for j, model_name in enumerate(MODEL_FOLDER_NAMES):
        row_cols = st.columns([1.5] + [1] * num_images_in_each_model)

        display_name = model_display_map.get(model_name, model_name)
        with row_cols[0]:
            st.write("")
            st.write("")
            st.subheader(f"Model: {display_name}")

        try:
            current_model_images = model_images_data[model_name]
            if len(current_model_images) != num_images_in_each_model:
                st.warning(f"모델 '{display_name}'의 이미지 개수가 예상치와 다릅니다. ({len(current_model_images)}개)")

            for i in range(num_images_in_each_model):
                with row_cols[i + 1]:
                    if i < len(current_model_images):
                        img_url = resolve_image_path(current_model_images[i])
                        if img_url:
                            st.image(img_url, width=300)
                        else:
                            st.error("이미지 경로 오류")
                    else:
                        st.empty()
        except KeyError:
            with row_cols[1]:
                st.error(f"모델 '{display_name}'의 이미지를 로드할 수 없습니다.")

    st.divider()
    st.subheader("평가")


    def _on_vote_change(rec_id, prompt, csv_path, keys, confirm_key, current_username):
        if not current_username:
            st.warning("사용자 이름을 입력해야 저장이 가능합니다.", icon="⚠️")
            return

        current_votes = {
            "consistency": st.session_state.get(keys["con"]),
            "alignment": st.session_state.get(keys["align"]),
            "quality": st.session_state.get(keys["qual"]),
        }
        try:
            df = upsert_votes(csv_path, rec_id, prompt, current_votes, current_username)
            st.session_state["votes"] = df

            if all(current_votes.values()):
                st.session_state[confirm_key] = True
                st.toast(f"저장됨: id={rec_id}")
            else:
                st.session_state[confirm_key] = False
        except Exception as e:
            st.error(f"저장 실패: {e}")


    vote_keys = {"con": vote_key_con, "align": vote_key_align, "qual": vote_key_qual}
    on_change_args = (rec_id, prompt, csv_path, vote_keys, confirm_key, username)

    vote_col1, vote_col2, vote_col3 = st.columns(3)
    with vote_col1:
        st.radio(
            "**1. Subject 일관성 (Consistency)**",
            options=vote_options,
            key=vote_key_con,
            format_func=format_model_name,
            index=vote_options.index(st.session_state[vote_key_con]) if st.session_state[
                                                                            vote_key_con] in vote_options else None,
            on_change=_on_vote_change, args=on_change_args,
        )
    with vote_col2:
        st.radio(
            "**2. text - image 일치도 (Alignment)**",
            options=vote_options,
            key=vote_key_align,
            format_func=format_model_name,
            index=vote_options.index(st.session_state[vote_key_align]) if st.session_state[
                                                                              vote_key_align] in vote_options else None,
            on_change=_on_vote_change, args=on_change_args,
        )
    with vote_col3:
        st.radio(
            "**3. 사실적인 이미지**",
            options=vote_options,
            key=vote_key_qual,
            format_func=format_model_name,
            index=vote_options.index(st.session_state[vote_key_qual]) if st.session_state[
                                                                             vote_key_qual] in vote_options else None,
            on_change=_on_vote_change, args=on_change_args,
        )

    st.divider()
    left_nav, mid_nav, right_nav = st.columns([1, 2, 1])
    with left_nav:
        if st.button("◀ 이전", use_container_width=True):
            st.session_state["idx"] = max(0, st.session_state["idx"] - 1)
            st.rerun()

    with right_nav:
        is_last_item = (st.session_state["idx"] + 1 == len(records))
        button_text = "평가 완료" if is_last_item else "다음 ▶"

        if st.button(button_text, use_container_width=True, type="primary"):
            if not st.session_state.get(confirm_key, False):
                st.warning("3가지 기준을 모두 선택(투표)해야 다음으로 진행할 수 있습니다.", icon="⚠️")
            else:
                if is_last_item:
                    st.balloons()
                    st.session_state['study_complete'] = True
                    st.rerun()
                else:
                    st.session_state["idx"] = min(len(records) - 1, st.session_state["idx"] + 1)
                    st.rerun()

    with mid_nav:
        st.markdown(
            f"<div style='text-align:center;'>항목 {st.session_state['idx'] + 1} / {len(records)}</div>",
            unsafe_allow_html=True
        )

elif st.session_state['study_complete']:

    # ====== 8. 평가 완료 페이지 (기존 로직 유지) ======
    st.title("🎉 평가가 완료되었습니다! 🎉")
    st.success("모든 평가 항목에 응답해주셔서 감사합니다.")

    st.divider()
    st.subheader(f"📊 {username}님의 투표 결과 집계")

    valid_vote_options = MODEL_FOLDER_NAMES
    df_all = st.session_state["votes"]  # 세션 상태의 votes DataFrame 사용

    if df_all.empty:
        st.error("투표 기록을 찾을 수 없습니다.")
    else:
        try:
            total_votes = len(df_all)
            st.metric(f"총 1명 참여", f"{total_votes}개 투표 (프롬프트 세트 기준)")

            votes_con_pct = pd.Series(dtype=float)
            votes_con_count = pd.Series(dtype=int)
            if "vote_consistency" in df_all:
                filtered_con = df_all[df_all['vote_consistency'].isin(valid_vote_options)]
                if not filtered_con.empty:
                    votes_con_pct = filtered_con['vote_consistency'].value_counts(normalize=True).mul(100)
                    votes_con_count = filtered_con['vote_consistency'].value_counts(normalize=False)

            votes_align_pct = pd.Series(dtype=float)
            votes_align_count = pd.Series(dtype=int)
            if "vote_alignment" in df_all:
                filtered_align = df_all[df_all['vote_alignment'].isin(valid_vote_options)]
                if not filtered_align.empty:
                    votes_align_pct = filtered_align['vote_alignment'].value_counts(normalize=True).mul(100)
                    votes_align_count = filtered_align['vote_alignment'].value_counts(normalize=False)

            votes_qual_pct = pd.Series(dtype=float)
            votes_qual_count = pd.Series(dtype=int)
            if "vote_quality" in df_all:
                filtered_qual = df_all[df_all['vote_quality'].isin(valid_vote_options)]
                if not filtered_qual.empty:
                    votes_qual_pct = filtered_qual['vote_quality'].value_counts(normalize=True).mul(100)
                    votes_qual_count = filtered_qual['vote_quality'].value_counts(normalize=False)

            res_col1, res_col2, res_col3 = st.columns(3)
            with res_col1:
                st.write("**1. Subject 일관성**")
                if not votes_con_pct.empty:
                    st.bar_chart(votes_con_pct)
                    df_con_summary = pd.concat([
                        votes_con_pct.rename("Percentage"),
                        votes_con_count.rename("Count")
                    ], axis=1).fillna(0)
                    st.dataframe(df_con_summary.reset_index(), use_container_width=True, hide_index=True)
                else:
                    st.caption("데이터 없음")
            with res_col2:
                st.write("**2. Prompt 일치도**")
                if not votes_align_pct.empty:
                    st.bar_chart(votes_align_pct)
                    df_align_summary = pd.concat([
                        votes_align_pct.rename("Percentage"),
                        votes_align_count.rename("Count")
                    ], axis=1).fillna(0)
                    st.dataframe(df_align_summary.reset_index(), use_container_width=True, hide_index=True)
                else:
                    st.caption("데이터 없음")
            with res_col3:
                st.write("**3. 이미지 품질**")
                if not votes_qual_pct.empty:
                    st.bar_chart(votes_qual_pct)
                    df_qual_summary = pd.concat([
                        votes_qual_pct.rename("Percentage"),
                        votes_qual_count.rename("Count")
                    ], axis=1).fillna(0)
                    st.dataframe(df_qual_summary.reset_index(), use_container_width=True, hide_index=True)
                else:
                    st.caption("데이터 없음")

            summary_df = pd.concat([
                votes_con_pct.rename('Consistency (%)'),
                votes_con_count.rename('Consistency (Count)'),
                votes_align_pct.rename('Alignment (%)'),
                votes_align_count.rename('Alignment (Count)'),
                votes_qual_pct.rename('Quality (%)'),
                votes_qual_count.rename('Quality (Count)')
            ], axis=1).fillna(0)
            summary_df.index.name = "Model"

            st.divider()
            st.subheader("결과 저장")

            summary_save_path = csv_path_for(STUDY_NAME, username).replace(f"_{username}.csv",
                                                                           f"_summary_{username}.csv")
            user_progress_file_path = csv_path_for(STUDY_NAME, username)
            button_text = f"💾 {username}님 집계 결과 저장 및 평가 기록 초기화"

            if st.button(button_text, type="primary", use_container_width=True):
                try:
                    summary_df.to_csv(summary_save_path, index=True, encoding='utf-8-sig')
                    st.success(f"성공! {username}님의 집계 결과가 서버에 저장되었습니다:\n{summary_save_path}")

                    if os.path.exists(user_progress_file_path):
                        os.remove(user_progress_file_path)
                        st.success(f"성공! '{username}'님의 평가 기록이 초기화되었습니다.")
                        st.warning("새 평가를 시작하려면 사이드바에서 '평가 시작 / 재시작' 버튼을 누르세요.")
                    else:
                        st.warning(f"'{username}'님의 평가 기록 파일을 찾지 못했습니다. (이미 초기화되었을 수 있습니다)")

                except Exception as e:
                    st.error(f"저장 또는 초기화 중 오류 발생: {e}")

            st.subheader("개인PC로 다운로드")

            raw_votes_csv = df_all.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "📝 개인 평가 기록 다운로드 (.csv)",
                raw_votes_csv,
                f"raw_votes_{STUDY_NAME}_{username}.csv",
                'text/csv',
                use_container_width=True
            )

            summary_csv = summary_df.to_csv(index=True, encoding='utf-8-sig').encode('utf-8-sig')
            st.download_button(
                "📊 집계 요약 결과 다운로드 (.csv)",
                summary_csv,
                f"summary_results_{STUDY_NAME}_{username}.csv",
                'text/csv',
                use_container_width=True
            )

        except Exception as e:
            st.error(f"결과 집계 중 오류 발생: {e}")
