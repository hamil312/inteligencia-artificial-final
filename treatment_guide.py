TREATMENT_GUIDE = {
    'nevus_comun': {
        'nombre': 'Nevus Común / Lunar Típico',
        'tipo': 'Benigno',
        'descripcion': (
            'Lesión melanocítica benigna, generalmente simétrica, '
            'bordes regulares, color homogéneo (marrón claro a oscuro). '
            'Presente desde la infancia o adolescencia.'
        ),
        'conducta': {
            'prioridad': 'Ninguna',
            'recomendacion': (
                'No requiere tratamiento. Se recomienda autoexamen mensual '
                'y control dermatológico anual de rutina.'
            ),
            'derivacion': 'No requiere',
        },
        'tratamiento': {
            'observacion': (
                'Seguimiento clínico. Fotografía corporal total (mapeo de lunares) '
                'si hay múltiples nevus.'
            ),
            'quirurgico': (
                'Solo por razones estéticas o si presenta cambios '
                '(crecimiento, asimetría, cambio de color). Extirpación '
                'con márgenes estrechos.'
            ),
        },
        'signos_alarma': (
            'Regla ABCDE: Asimetría, Bordes irregulares, Color variado, '
            'Diámetro >6mm, Evolución/cambio. Ante cualquiera de estos, '
            'consulta URGENTE al dermatólogo.'
        ),
        'pronostico': 'Excelente. Sin riesgo de malignización en ausencia de cambios.',
    },
    'nevus_atipico': {
        'nombre': 'Nevus Atípico / Lunar Displásico',
        'tipo': 'Benigno con atipia',
        'descripcion': (
            'Lesión melanocítica con características clínicas atípicas: '
            'asimetría leve, bordes difusos, coloración variable. '
            'Representa un factor de riesgo para melanoma, especialmente '
            'en pacientes con múltiples nevus atípicos (síndrome del nevus atípico).'
        ),
        'conducta': {
            'prioridad': 'Vigilancia estrecha',
            'recomendacion': (
                'Control dermatológico cada 3-6 meses con mapeo corporal total. '
                'Autoexamen mensual. Evitar exposición solar excesiva y uso '
                'de fotoprotección SPF50+ diario.'
            ),
            'derivacion': 'Dermatólogo para seguimiento periódico',
        },
        'tratamiento': {
            'observacion': (
                'Fotografía clínica seriada y dermatoscopia digital de seguimiento. '
                'Documentar cambios en el tiempo.'
            ),
            'quirurgico': (
                'Biopsia escisional indicada si hay cambios documentados '
                'en el seguimiento o criterios dermatoscópicos de sospecha. '
                'Margen de 1-2mm. Estudio histopatológico completo.'
            ),
        },
        'signos_alarma': (
            'Cambio en tamaño, forma o color. Picazón, sangrado o ulceración. '
            'Aparición de un área pigmentada alrededor del nevus (halo). '
            'Cualquier cambio requiere evaluación dermatológica URGENTE.'
        ),
        'pronostico': (
            'Bueno con vigilancia adecuada. El riesgo de transformación '
            'a melanoma es bajo pero real (estimado 1-5% en pacientes '
            'con múltiples nevus atípicos).'
        ),
    },
    'melanoma': {
        'nombre': 'Melanoma Maligno',
        'tipo': 'Maligno',
        'descripcion': (
            'Neoplasia maligna derivada de melanocitos. Es el cáncer de piel '
            'más agresivo pero curable si se detecta en etapas tempranas. '
            'Puede presentarse como lesión nueva o cambios en un lunar existente. '
            'Clasificación: melanoma de extensión superficial (más común), '
            'nodular, lentigo maligno, acral lentiginoso.'
        ),
        'conducta': {
            'prioridad': 'URGENTE — Atención inmediata',
            'recomendacion': (
                'SOSPECHA DE MELANOMA. Derivación URGENTE a dermatología '
                'para biopsia escisional diagnóstica. No demorar más de 2-4 semanas. '
                'Evitar manipulación o traumatismo de la lesión.'
            ),
            'derivacion': 'URGENTE — Dermatólogo / Cirugía dermatológica',
        },
        'tratamiento': {
            'quirurgico': (
                'Estándar de oro: escisión quirúrgica con márgenes según '
                'espesor de Breslow: '
                '- In situ: margen 0.5cm '
                '- Breslow ≤1mm: margen 1cm '
                '- Breslow 1-2mm: margen 1-2cm '
                '- Breslow >2mm: margen 2cm. '
                'Estudio de ganglio centinela si Breslow >0.8mm o con ulceración.'
            ),
            'adyuvante': (
                'Según etapa: inmunoterapia (anti-PD1: nivolumab, pembrolizumab), '
                'terapia dirigida (BRAF/MEK inhibidores si mutación BRAF V600), '
                'interferón-alfa. Evaluación por oncología.'
            ),
            'metastasico': (
                'Inmunoterapia combinada (nivolumab+ipilimumab) o terapia '
                'dirigida BRAF/MEK. Radioterapia paliativa para metástasis '
                'sintomáticas. Ensayos clínicos disponibles.'
            ),
        },
        'signos_alarma': (
            'ABCDE: Asimetría, Bordes irregulares, Coloración múltiple, '
            'Diámetro >6mm, Evolución. EFG: Elevación, Firmeza, '
            'Crecimiento progresivo. PLUS: sangrado, ulceración, prurito.'
        ),
        'pronostico': (
            'Depende del espesor de Breslow al diagnóstico: '
            '- <0.8mm sin ulceración: supervivencia 5 años >95% '
            '- 0.8-1mm: ~90% '
            '- 1-2mm: ~75-85% '
            '- 2-4mm: ~60-70% '
            '- >4mm: ~40-50% '
            'Metastásico: ~25-30% a 5 años con inmunoterapia actual. '
            'La detección temprana es el factor pronóstico más importante.'
        ),
    },
}


def get_guide(class_name):
    return TREATMENT_GUIDE.get(class_name)


def format_guide(class_name):
    guide = get_guide(class_name)
    if not guide:
        return "Información no disponible."

    lines = []
    lines.append(f"### {guide['nombre']}")
    lines.append(f"**Tipo:** {guide['tipo']}")
    lines.append("")
    lines.append("**Descripción:**")
    lines.append(guide['descripcion'])
    lines.append("")
    lines.append(f"**Prioridad:** {guide['conducta']['prioridad']}")
    lines.append("")
    lines.append("**Recomendación:**")
    lines.append(guide['conducta']['recomendacion'])
    lines.append("")
    lines.append(f"**Derivación:** {guide['conducta']['derivacion']}")
    lines.append("")

    for key, val in guide['tratamiento'].items():
        label = key.replace('_', ' ').title()
        lines.append(f"**{label}:**")
        lines.append(val)
        lines.append("")

    lines.append("**Signos de alarma:**")
    lines.append(guide['signos_alarma'])
    lines.append("")
    lines.append("**Pronóstico:**")
    lines.append(guide['pronostico'])
    lines.append("")
    lines.append("---")
    lines.append("*Esta guía es informativa. Consulte a un dermatólogo.*")

    return "\n".join(lines)
