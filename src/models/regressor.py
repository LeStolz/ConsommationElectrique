from abc import ABC, abstractmethod
import pandas as pd


class Regressor(ABC):
    """
    Interface commune pour tous les modèles de prévision temporelle.
    """

    @abstractmethod
    def fit(self, df_train: pd.DataFrame):
        """
        Entraîne le modèle ou mémorise l'historique nécessaire à la prédiction.
        """
        pass


    @abstractmethod
    def predict(self, df_test: pd.DataFrame):
        """
        Génère les prédictions pour la période de test.
        Doit retourner une Pandas Series ou un array de la même taille que df_test.
        """
        pass