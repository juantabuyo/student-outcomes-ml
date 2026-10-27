"""Experiment settings. Source column names remain unchanged for traceability."""

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
N_JOBS = 1  # Avoid nested parallelism; deterministic runs also use one BLAS thread.
SEPARATOR = ";"
CLASSIFICATION_TARGET = 'objetivo'
REGRESSION_TARGET = 'nota_media_2sem'
BINARY_COLUMNS = ['desplazado', 'necesidades_educativas_especiales', 'deudor', 'matricula_al_dia', 'becado', 'internacional']
CATEGORICAL_COLUMNS = ['estado_civil', 'modo_solicitud', 'curso', 'asistencia_diurna_vespertina', 'cualificacion_previa', 'nacionalidad', 'cualificacion_madre', 'cualificacion_padre', 'ocupacion_madre', 'ocupacion_padre', 'genero']
SECOND_SEMESTER_COLUMNS = ['asignaturas_2sem_convalidadas', 'asignaturas_2sem_matriculadas', 'asignaturas_2sem_evaluadas', 'asignaturas_2sem_aprobadas', 'nota_media_2sem', 'asignaturas_2sem_sin_evaluacion']

# Predictors with no dedicated role above. With every column named above they make up the 37
# columns of data_schema.csv, which is the whole input contract: load_data rejects any other.
PASS_THROUGH_COLUMNS = ['asignaturas_1sem_convalidadas']

CLASS_LABELS = {"abandono": "Dropout", "matriculado": "Enrolled", "graduado": "Graduate"}
FIRST_SEMESTER_NUMERIC = ['orden_solicitud', 'nota_cualificacion_previa', 'nota_admision', 'edad_al_matricularse', 'asignaturas_1sem_matriculadas', 'asignaturas_1sem_evaluadas', 'asignaturas_1sem_aprobadas', 'nota_media_1sem', 'asignaturas_1sem_sin_evaluacion', 'tasa_desempleo', 'tasa_inflacion', 'pib']
SECOND_SEMESTER_NUMERIC = ['asignaturas_2sem_matriculadas', 'asignaturas_2sem_evaluadas', 'asignaturas_2sem_aprobadas', 'nota_media_2sem', 'asignaturas_2sem_sin_evaluacion']

# Presentation only: source columns and stored CSV keys are never renamed.
FEATURE_LABELS = {
    'estado_civil': 'Marital status', 'modo_solicitud': 'Application mode',
    'orden_solicitud': 'Application order', 'curso': 'Course',
    'asistencia_diurna_vespertina': 'Daytime/evening attendance',
    'cualificacion_previa': 'Previous qualification',
    'nota_cualificacion_previa': 'Previous qualification grade',
    'nacionalidad': 'Nationality', 'cualificacion_madre': "Mother's qualification",
    'cualificacion_padre': "Father's qualification", 'ocupacion_madre': "Mother's occupation",
    'ocupacion_padre': "Father's occupation", 'nota_admision': 'Admission grade',
    'desplazado': 'Displaced', 'necesidades_educativas_especiales': 'Special educational needs',
    'deudor': 'Debtor', 'matricula_al_dia': 'Tuition fees up to date',
    'genero': 'Gender', 'becado': 'Scholarship holder',
    'edad_al_matricularse': 'Age at enrolment', 'internacional': 'International student',
    'tasa_desempleo': 'Unemployment rate', 'tasa_inflacion': 'Inflation rate', 'pib': 'GDP',
    'objetivo': 'Final outcome',
}
for _semester in [1, 2]:
    FEATURE_LABELS[f'nota_media_{_semester}sem'] = f'Semester {_semester}: average grade'
    for _source, _display in [('convalidadas', 'credited'), ('matriculadas', 'enrolled'),
                             ('evaluadas', 'evaluations'), ('aprobadas', 'approved'),
                             ('sin_evaluacion', 'without evaluations')]:
        FEATURE_LABELS[f'asignaturas_{_semester}sem_{_source}'] = f'Semester {_semester}: units {_display}'


def feature_label(source_name):
    """Display names only; dummy-category suffixes keep source category values."""
    if source_name in FEATURE_LABELS:
        return FEATURE_LABELS[source_name]
    for column in sorted(CATEGORICAL_COLUMNS, key=len, reverse=True):
        if source_name.startswith(column + '_'):
            return FEATURE_LABELS.get(column, column) + ': ' + source_name[len(column) + 1:]
    return source_name
