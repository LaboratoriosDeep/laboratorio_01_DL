import os
import pandas as pd

from config import NON_FEATURE_COLS

def load_data(filepath='data/raw/15_atributos_R0-R5.sav'):
    """
    Carga el dataset desde un archivo SPSS (.sav) utilizando pandas y pyreadstat.
    """
    # Ajustar la ruta si se ejecuta desde dentro de la carpeta src/
    if not os.path.exists(filepath) and os.path.exists('../' + filepath):
        filepath = '../' + filepath
        
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró el archivo en: {filepath}")

    try:
        df = pd.read_spss(filepath)
        print(f"[data_loader] Dataset cargado correctamente. Dimensiones: {df.shape}")
        
        return df
    except Exception as e:
        print(f"[data_loader] Error al cargar el archivo: {e}")
        return None
    
def get_X_y(df, target_col='GDS'):

    # Se excluye el ID y TODAS las variantes de GDS para no hacer "trampa" en la predicción.
    columnas_excluir = NON_FEATURE_COLS
    
    # X: Variables predictoras (aseguramos que sean numéricas para los modelos)
    X = df.drop(columns=columnas_excluir, errors='ignore').astype(float)
    
    # y: Variable objetivo
    y = df[target_col].astype(int)
    
    return X, y

def verify_integrity(X):
    """
    Verifica que no haya nulos y que los predictores sean todos binarios.
    """
    missing = X.isnull().sum().sum()
    if missing > 0:
        print(f"[data_loader] ADVERTENCIA: Se encontraron {missing} valores nulos en X.")
    else:
        print("[data_loader] Sin valores nulos en los predictores.")

    # Verificación de binariedad extraída del notebook del profesor
    is_binary = X.isin([0.0, 1.0]).all().all()
    if not is_binary:
        print("[data_loader] ADVERTENCIA: Hay columnas con valores no binarios.")
    else:
        print("[data_loader] Todos los predictores son binarios.")

# ── Prueba rápida ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    ruta = 'data/raw/15_atributos_R0-R5.sav' 
    
    df_raw = load_data(ruta)
    if df_raw is not None:
        X_data, y_data = get_X_y(df_raw, target_col='GDS')
        verify_integrity(X_data)
        print(f"\nResumen: X shape: {X_data.shape}, y shape: {y_data.shape}")
        print(f"Distribución de clases en 'GDS':\n{y_data.value_counts().sort_index()}")