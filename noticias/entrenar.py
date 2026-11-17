import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

df = pd.read_csv("datos.csv").dropna(subset=["texto", "etiqueta"])
X_train, X_test, y_train, y_test = train_test_split(
    df["texto"].astype(str), df["etiqueta"].astype(int),
    test_size=0.2, random_state=42, stratify=df["etiqueta"]
)

modelo = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2), max_df=0.9, min_df=2, sublinear_tf=True),
    LogisticRegression(max_iter=1000),
)
modelo.fit(X_train, y_train)
print(f"Precisión en prueba: {modelo.score(X_test, y_test):.2%}")

joblib.dump(modelo, "modelo_noticias.joblib")
print("Modelo guardado en modelo_noticias.joblib")