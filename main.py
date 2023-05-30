from streamlit import button, cache_resource, columns, markdown, plotly_chart, radio, selectbox, slider
from datetime import date
from joblib import load
from pandas import DataFrame
from plotly.subplots import make_subplots
from plotly.graph_objects import Bar
from sklearn.base import BaseEstimator, TransformerMixin


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, include_all: bool, include_bias: bool) -> None:
        super().__init__()
        self.include_all = include_all
        self.include_bias = include_bias

    def fit(self, x, y=None):
        return self

    def transform(self, x, y = None):
        x['OcupacionEconomica'].replace(
            'No Aplica', '<No Registra>', inplace=True)
        x['Posee Cónyuge o Compañero(a)?'].replace(
            '<No Registra>', '<No Aplica>', inplace=True)
        x['Línea de FpT para el Máx. Nivel'].replace(
            '<No Registra>', '<No Aplica>', inplace=True)
        if not self.include_all:
            minimum = ['Estado de la vinculación ASS', 'Posee Cónyuge o Compañero(a)?',
                       'Línea de FpT para el Máx. Nivel', 'Desembolso BIE', 'OcupacionEconomica']
            x.drop(
                columns=[col for col in x.columns if col not in minimum], inplace=True)
            return x
        x['Máximo Nivel FpT Reportado'].replace(
            'Técnico Laboral', 'Técnico', inplace=True)
        x['Máximo Nivel FpT Reportado'].replace(
            'Técnico Profesional', 'Técnico', inplace=True)
        x['Máximo Nivel FpT Reportado'].replace(
            'Técnico Laboral por Competencias', 'Técnico', inplace=True)
        x['Máximo Nivel FpT Reportado'].replace(
            'Especialización Técnica', 'Técnico', inplace=True)
        x['Máximo Nivel FpT Reportado'].replace(
            'Especialización Tecnológica', 'Tecnológico', inplace=True)
        x['Máximo Nivel FpT Reportado'].replace(
            'Operario', 'Otro', inplace=True)
        x['Máximo Nivel FpT Reportado'].replace(
            'Auxiliar', 'Otro', inplace=True)
        x['Máximo Nivel FpT Reportado'].replace(
            'Certificación por Evaluación de Competencias', 'Otro', inplace=True)
        x['Grupo Etario'].replace(
            'Entre 18 y 25 años', 'Entre 18 y 40 años', inplace=True)
        x['Grupo Etario'].replace(
            'Entre 26 y 40 años', 'Entre 18 y 40 años', inplace=True)
        x['Régimen de tenencia Vivienda'].replace(
            'Propia, totalmente pagada', 'Propia', inplace=True)
        x['Régimen de tenencia Vivienda'].replace(
            'Propia, la están pagando', 'Propia', inplace=True)
        x['Régimen de tenencia Vivienda'].replace(
            'Sana posesión con título', 'Propia', inplace=True)
        x['Régimen de tenencia Vivienda'].replace(
            'Es usufructo', 'Con permiso del propietario, sin pago alguno', inplace=True)
        x['Régimen de tenencia Vivienda'].replace(
            'Familiar', 'Con permiso del propietario, sin pago alguno', inplace=True)
        x['Régimen de tenencia Vivienda'].replace(
            'Posesión sin título (ocupante de hecho) o propiedad colectiva',
            'Otra', inplace=True)
        x['Régimen de tenencia Vivienda'].replace(
            'Otra forma de tenencia  (posesión sin título, ocupante de hecho, propiedad colectiva, etc)',
            'Otra', inplace=True)
        x['Tipo de Vivienda'].replace(
            'Casa-Lote', 'Casa', inplace=True)
        x['Tipo de Vivienda'].replace(
            'Cuarto(s)', 'Habitación', inplace=True)
        x['Tipo de Vivienda'].replace(
            'Rancho', 'Finca', inplace=True)
        x['Tipo de Vivienda'].replace(
            'Vivienda (casa) indígena', 'Casa', inplace=True)
        x['Tipo de Vivienda'].replace(
            'Otro tipo de vivienda (carpa, tienda, vagón, embarcación, cueva, refugio natural, puente, calle, etc.)',
            'Otro', inplace=True)
        x['Sexo'] = x['Sexo'].str.upper()
        x['N° de Hijos'].replace(-2, -1, inplace=True)
        exclude = ['Ex Grupo', 'Año desmovilización', 'Ingresó/No ingresó', 'Año de Independización/Ingreso',
                   'Departamento de residencia', 'Municipio de residencia', 'BeneficioTRV', 'BeneficioFA', 'BeneficioFPT', 'BeneficioPDT',
                   'Tipo de BIE Accedido', 'Estado ISUN', 'Posee Servicio Social?', 'Posee Censo de Familia?',
                   'Posee Censo de Habitabilidad?', 'Clasificación Componente Específico', 'FechaCorte', 'FechaActualizacion']
        if not self.include_bias:
            exclude.append('DesagregadoDesembolsoBIE')
        x.drop(
            columns=[col for col in x.columns if col in exclude], inplace=True)
        return x


@cache_resource
def load_models():
    return {
        'Simple': load(f'./models/tree_unbiased.pkl'),
        'Completo': load(f'./models/tree_full_unbiased.pkl')
    }


models = load_models()


def predict(data, model_name):
    classes = {
        0: 'Culminará el proceso',
        1: 'Abandonará el proceso'
    }
    model = models[model_name]
    fig = make_subplots()
    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)')
    probability = model.predict_proba(data)
    prediction = (probability.copy()[:, 1] >= 0.1).astype(int)[0]
    markdown(f'La clasificación es  *{classes[prediction]}*')
    fig.add_trace(
        Bar(
            x=list(classes.values()),
            y=probability[0],
            name='Árbol completo'
        ))
    fig.update_yaxes(range=[0, 1])
    fig.update_xaxes(title_text='Clase')
    plotly_chart(fig, use_container_width=True)


today = date.today()
Width = 700
H = 600
markdown("<h1 style='text-align: center'>Análisis de Factores de Riesgo que influyen en la deserción de desmovilizados que han ingresado al proceso de reintegración</h1>", True)
markdown("<h2 style='text-align: center'>Oscar Julián Castañeda & Juan David Ayala</h5>", True)
modelo = radio('¿Que modelo quiere utilizar?', ('Simple', 'Completo'))

linea_fpt_options = (
    '<No Aplica>', 'OTROS',
    'OPERADORES DE MAQUINAS, EQUIPO Y TRANSPORTE', 'SERVICIOS',
    'CARPINTERIA Y EBANISTERIA', 'SALUD',
    'MECANICA AUTOMOTRIZ Y DE MOTOS', 'SISTEMAS',
    'FINANZAS Y ADMINISTRACION', 'AGROPECUARIA', 'ALIMENTOS Y BEBIDAS',
    'ELECTRICIDAD', 'CONSTRUCCION', 'MERCADEO Y VENTAS',
    'MECANICA INDUSTRIAL', 'TRANSVERSAL', 'ELECTRONICA',
    'CONFECCION, MARROQUINERIA Y CALZADO', 'AMBIENTAL',
    'ARTESANIAS Y JOYERIA', 'ESTETICA',
    'EXPLOTACION MINERA, PETROLEO Y GAS',
    'DISEÑO Y ARTES GRAFICAS')
ocupacion_options = (
    'Ocupados en el sector Informal',
    '<No Registra>',
    'Población Económicamente Inactiva',
    'Desocupados')
desembolso_bie_options = ('Sí', 'No')
estado_ass_options = (
    '<No Aplica>',
    'Certificado',
    'Abandono sin justa causa',
    'Abandono con justa causa',
    'Vinculado')
posee_conyugue_options = ('<No Aplica>', 'No', 'Sí')

if modelo == 'Simple':
    inputs = columns(2)
    with inputs[0]:
        linea_fpt = selectbox(
            'Línea de FpT para el Máximo Nivel', linea_fpt_options)
        desembolso_bie = selectbox(
            '¿Desembolsó el BIE?', desembolso_bie_options)
        posee_conyugue = selectbox(
            'Posee Cónyuge o Compañero(a)?', posee_conyugue_options)
    with inputs[1]:
        ocupacion = selectbox('Ocupacion económica', ocupacion_options)
        estado_ass = selectbox(
            'Estado de la vinculación ASS', estado_ass_options)
    data = DataFrame({
        'Línea de FpT para el Máx. Nivel': [linea_fpt],
        'OcupacionEconomica': [ocupacion],
        'Desembolso BIE': [desembolso_bie],
        'Estado de la vinculación ASS': [estado_ass],
        'Posee Cónyuge o Compañero(a)?': [posee_conyugue],
    })
else:
    inputs = columns(3)
    with inputs[0]:
        tipo_desmovilizacion = selectbox('Tipo de Desmovilización', (
            'Colectiva',
            'Individual'))
        nivel_educativo = selectbox('Nivel Educativo', (
            'Alfabetización',
            'Bachiller',
            'Básica Primaria',
            'Básica Secundaria',
            'Por Establecer'))
        ocupacion = selectbox('Ocupacion económica', ocupacion_options)
        tipo_ass = selectbox('Tipo de ASS Vinculada', (
            '<No Aplica>',
            'Acompañamiento a la atención en Salud y atención Alimentaria a comunidades vulnerables',
            'Embellecimiento de Espacio Publico',
            'Aporte de habilidades Especiales que le participante ponga a disposición de la comunidad',
            'Multiplicadores del Conocimiento',
            'Generación de espacios de recreación, Arte, Cultura y Deporte',
            'Recuperación Ambiental'))
        regimen_vivienda = selectbox('Régimen de tenencia Vivienda', (
            '<No Aplica>',
            'Con permiso del propietario, sin pago alguno',
            'En arriendo o subarriendo', 'Es usufructo',
            'Familiar', 'Otra forma de tenencia (posesión sin título, ocupante de hecho, propiedad colectiva, etc)',
            'Posesión sin título (ocupante de hecho) o propiedad colectiva',
            'Propia, la están pagando',
            'Propia, totalmente pagada',
            'Sana posesión con título'))

    with inputs[1]:
        grup_etario = selectbox('Grupo Etario', (
            'Entre 18 y 25 años',
            'Entre 26 y 40 años',
            'Entre 41 y 60 años',
            'Mayor de 60 años'))
        maximo_fpt = selectbox('Máximo Nivel FpT Reportado', (
            '<No Aplica>',
            'Complementario',
            'Técnico',
            'Semicalificado',
            'Tecnológico',
            'Operario',
            'Transversal',
            'Técnico Profesional',
            'Técnico Laboral',
            'Auxiliar',
            'Técnico Laboral por Competencias',
            'Especialización Tecnológica',
            'Certificación por Evaluación de Competencias',
            'Especialización Técnica'
        ))
        desembolso_bie = selectbox(
            '¿Desembolsó el BIE?', desembolso_bie_options)
        posee_conyugue = selectbox(
            'Posee Cónyuge o Compañero(a)?', posee_conyugue_options)
        servicios_publicos = selectbox('Posee Serv. Públicos Básicos',
                                       ('<No Aplica>', 'No', 'Sí'))

    with inputs[2]:
        sexo = selectbox('Sexo', ('Masculino', 'Femenino'))
        linea_fpt = selectbox(
            'Línea de FpT para el Máximo Nivel', linea_fpt_options)
        estado_ass = selectbox(
            'Estado de la vinculación ASS', estado_ass_options)
        tipo_vivienda = selectbox('Tipo de Vivienda', (
            '<No Aplica>', 'Casa', 'Apartamento', 'Casa-Lote', 'Habitación',
            'Finca', 'Rancho', 'Otro', 'Cuarto(s)', 'Vivienda (casa) indígena',
            'Otro tipo de vivienda (carpa, tienda, vagón, embarcación, cueva, refugio natural, puente, calle, etc.)'))
        regimen_salud = selectbox('Régimen de salud', (
            'S - SUBSIDIADO',
            '<No Registra>',
            'C - CONTRIBUTIVO'))
    numeric_inputs = columns([1, 2])
    with numeric_inputs[0]:
        val_hijos = radio('¿Tiene registro de hijos?', ('Sí', 'No'))
        if val_hijos == 'Sí':
            with numeric_inputs[1]:
                hijos = slider('Número de hijos ', 1, 10, (1))
        else:
            hijos = -1
        val_familia = radio('¿Tiene registro de familia?', ('Sí', 'No'))
        if val_familia == 'Sí':
            with numeric_inputs[1]:
                familia = slider('Integrantes grupo familiar', 1, 20, (1))
        else:
            familia = -1
    data = DataFrame({
        'Tipo de Desmovilización': [tipo_desmovilizacion],
        'Grupo Etario': [grup_etario],
        'Sexo': [sexo],
        'Nivel Educativo': [nivel_educativo],
        'Máximo Nivel FpT Reportado': [maximo_fpt],
        'Línea de FpT para el Máx. Nivel': [linea_fpt],
        'OcupacionEconomica': [ocupacion],
        'Desembolso BIE': [desembolso_bie],
        'Estado de la vinculación ASS': [estado_ass],
        'Tipo de ASS Vinculada': [tipo_ass],
        'Posee Cónyuge o Compañero(a)?': [posee_conyugue],
        'N° de Hijos': [hijos],
        'Total Integrantes grupo familiar': [familia],
        'Tipo de Vivienda': [tipo_vivienda],
        'Régimen de tenencia Vivienda': [regimen_vivienda],
        'Posee Serv. Públicos Básicos': [servicios_publicos],
        'Régimen de salud': [regimen_salud]
    })
if (button('Realizar Clasificación')):
    predict(data, modelo)
