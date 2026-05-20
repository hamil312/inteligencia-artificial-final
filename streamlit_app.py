import os
import sys
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras import Model
from PIL import Image
import matplotlib.pyplot as plt
import io

from treatment_guide import format_guide
from llm_advisor import LLMAdvisor, ADVISOR_AVAILABLE, MODEL_OPTIONS, format_response

from skin_lesion_assistant import (
    build_irv2_sa_model, SoftAttention, CLASS_NAMES as CLASS_NAMES_IRV2,
    CLASS_DESCRIPTIONS as CLASS_DESC_IRV2, CLASS_SEVERITY as CLASS_SEV_IRV2
)

from CNNMejorado_PH2 import (
    build_cnn_mejorado_model, build_mobilenetv2_model,
    CLASS_NAMES as CLASS_NAMES_CNN,
    CLASS_DESCRIPTIONS as CLASS_DESC_CNN, CLASS_SEVERITY as CLASS_SEV_CNN,
    IMG_SIZE as CNN_IMG_SIZE
)

st.set_page_config(
    page_title="Clasificador de Lesiones Cutáneas",
    page_icon="🔬",
    layout="centered"
)

WEIGHTS_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CONFIGS = {
    'IRV2+SA': {
        'build_fn': lambda: build_irv2_sa_model(freeze_backbone=True),
        'weights': os.path.join(WEIGHTS_DIR, 'IRV2+SA_PH2.weights.h5'),
        'img_size': 299,
        'class_names': CLASS_NAMES_IRV2,
        'descriptions': CLASS_DESC_IRV2,
        'severity': CLASS_SEV_IRV2,
        'preprocess': tf.keras.applications.inception_resnet_v2.preprocess_input,
    },
    'CNN (desde cero)': {
        'build_fn': build_cnn_mejorado_model,
        'weights': os.path.join(WEIGHTS_DIR, 'cnn_mejorado_ph2.weights.h5'),
        'img_size': CNN_IMG_SIZE,
        'class_names': CLASS_NAMES_CNN,
        'descriptions': CLASS_DESC_CNN,
        'severity': CLASS_SEV_CNN,
        'preprocess': lambda x: (x / 255.0 - 0.5) / 0.5,
    },
    'MobileNetV2': {
        'build_fn': lambda: build_mobilenetv2_model()[0],
        'weights': os.path.join(WEIGHTS_DIR, 'mobilenetv2_ph2.weights.h5'),
        'img_size': CNN_IMG_SIZE,
        'class_names': CLASS_NAMES_CNN,
        'descriptions': CLASS_DESC_CNN,
        'severity': CLASS_SEV_CNN,
        'preprocess': tf.keras.applications.mobilenet_v2.preprocess_input,
    },
}


@st.cache_resource
def load_model(model_key):
    cfg = MODEL_CONFIGS[model_key]
    model = cfg['build_fn']()

    try:
        model.load_weights(cfg['weights'])
        st.success(f"Pesos cargados: {os.path.basename(cfg['weights'])}")
    except:
        st.warning(f"Pesos no encontrados: {cfg['weights']}")
    return model, cfg


def predict(model, cfg, img):
    img_resized = img.resize((cfg['img_size'], cfg['img_size']))
    img_array = np.array(img_resized, dtype=np.float32)
    img_processed = cfg['preprocess'](img_array)
    img_batch = np.expand_dims(img_processed, axis=0)
    preds = model.predict(img_batch, verbose=0)[0]
    return preds, img_resized


def plot_confidence(predictions, class_names, title=""):
    fig, ax = plt.subplots(figsize=(6, 3))
    colors = ['#e74c3c' if c == 'melanoma' else '#3498db' for c in class_names]
    bars = ax.barh(class_names, predictions, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confianza')
    if title:
        ax.set_title(title)
    for bar, p in zip(bars, predictions):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{p:.1%}', va='center', fontsize=10)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def main():
    st.title("🔬 Clasificador de Lesiones Cutáneas")
    st.markdown("Sube una imagen dermatoscópica para clasificarla.")

    col1, col2 = st.columns([1, 2])
    with col1:
        model_key = st.radio(
            "Modelo", list(MODEL_CONFIGS.keys()),
            help="IRV2+SA: mayor precisión (299×299) | CNN o MobileNetV2: más ligeros (64×64)"
        )
    with col2:
        uploaded = st.file_uploader(
            "Elige una imagen", type=['jpg', 'jpeg', 'png', 'bmp']
        )

    if not uploaded:
        st.info("Sube una imagen para comenzar.")
        st.markdown("---")
        st.markdown("**Clases disponibles:**")
        cfg = MODEL_CONFIGS[model_key]
        for c in cfg['class_names']:
            emoji = "🔴" if c == 'melanoma' else "🟢"
            st.markdown(f"{emoji} **{c}** — {cfg['descriptions'][c]}")
        return

    model, cfg = load_model(model_key)
    img = Image.open(uploaded).convert('RGB')

    col_img, col_res = st.columns([1, 1])
    with col_img:
        st.image(img, caption="Imagen original", width=250)

    predictions, img_resized = predict(model, cfg, img)
    idx = np.argmax(predictions)
    cls = cfg['class_names'][idx]
    conf = predictions[idx]

    with col_res:
        fig = plot_confidence(predictions, cfg['class_names'])
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    if cls == 'melanoma':
        st.error(f"### Diagnóstico: {cls.upper()}")
    else:
        st.info(f"### Diagnóstico: {cls.upper()}")

    colm1, colm2, colm3 = st.columns(3)
    colm1.metric("Confianza", f"{conf:.1%}")
    colm2.metric("Clase", cls.replace('_', ' ').title())
    colm3.metric("Severidad", cfg['severity'][cls])

    st.markdown(f"**Descripción:** {cfg['descriptions'][cls]}")

    for i in np.argsort(predictions)[::-1]:
        c = cfg['class_names'][i]
        st.markdown(f"- {c}: {predictions[i]:.1%}")

    severity_color = {
        'BAJA (benigno)': '🟢',
        'MEDIA (benigno con atipia, vigilancia)': '🟡',
        'ALTA (precanceroso)': '🟠',
        'CRÍTICA (maligno agresivo)': '🔴',
    }
    sev_emoji = severity_color.get(cfg['severity'][cls], '⚪')

    with st.expander(f"{sev_emoji} Guía de orientación — {cls.replace('_', ' ').title()}", expanded=True):
        st.markdown(format_guide(cfg['class_names'][idx]))

    with st.expander("🤖 Asesor LLM (experimental)", expanded=False):
        if not ADVISOR_AVAILABLE:
            st.warning(
                "Librería `transformers` no encontrada. "
                "Instálala con: `pip install transformers torch`"
            )
        else:
            if 'llm_loaded' not in st.session_state:
                st.session_state.llm_loaded = False
                st.session_state.llm_response = None

            col_llm1, col_llm2 = st.columns([1, 2])
            with col_llm1:
                model_choice = st.selectbox(
                    "Modelo LLM",
                    options=list(MODEL_OPTIONS.keys()) if ADVISOR_AVAILABLE else [],
                    key="llm_model_choice"
                )
            with col_llm2:
                if not st.session_state.llm_loaded:
                    if st.button("🔄 Cargar y consultar LLM"):
                        with st.spinner(f"Cargando {MODEL_OPTIONS[model_choice]}..."):
                            try:
                                advisor = LLMAdvisor(MODEL_OPTIONS[model_choice])
                                advisor.load()
                                st.session_state.advisor = advisor
                                st.session_state.llm_loaded = True
                            except Exception as e:
                                st.error(f"Error al cargar LLM: {e}")

            if st.session_state.llm_loaded:
                st.success(f"LLM cargado en {st.session_state.advisor.device}")
                if st.button("💬 Consultar al asistente"):
                    with st.spinner("Generando respuesta..."):
                        try:
                            resp = st.session_state.advisor.generate(
                                cls, conf, cfg['descriptions'][cls], cfg['severity'][cls]
                            )
                            st.session_state.llm_response = resp
                        except Exception as e:
                            st.error(f"Error: {e}")

                if st.session_state.llm_response:
                    st.markdown("---")
                    st.markdown(format_response(st.session_state.llm_response))
                    st.caption("⚠ Respuesta generada por IA. Verificar con un profesional.")

    st.markdown("---")
    with st.expander("ℹ️ Información del modelo", expanded=False):
        total = sum(tf.keras.backend.count_params(w) for w in model.weights)
        trainable = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
        st.markdown(f"**Arquitectura:** {model_key}")
        st.markdown(f"**Tamaño entrada:** {cfg['img_size']}×{cfg['img_size']}")
        st.markdown(f"**Parámetros:** {total:,} totales, {trainable:,} entrenables")
        st.markdown(f"**Dataset:** PH2 (200 imágenes, 3 clases)")

    if model_key == 'IRV2+SA' and st.button("Mostrar mapa de atención"):
        try:
            sa_model = Model(
                inputs=model.inputs,
                outputs=model.get_layer('soft_attention').output
            )
            _, sa_maps = sa_model.predict(np.expand_dims(
                cfg['preprocess'](np.array(img.resize((299, 299)), dtype=np.float32)),
                axis=0
            ), verbose=0)
            attn = np.sum(sa_maps[0], axis=0)

            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            ax1.imshow(img_resized)
            ax1.set_title("Original"); ax1.axis('off')
            ax2.imshow(attn, cmap='jet')
            ax2.set_title("Atención"); ax2.axis('off')
            ax3.imshow(img_resized, alpha=0.6)
            ax3.imshow(attn, cmap='jet', alpha=0.4)
            ax3.set_title("Superposición"); ax3.axis('off')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.error(f"No se pudo generar mapa de atención: {e}")

    st.markdown("---")
    st.caption("⚠ Referencia informativa. Consulte a un dermatólogo para un diagnóstico profesional.")


if __name__ == '__main__':
    main()
