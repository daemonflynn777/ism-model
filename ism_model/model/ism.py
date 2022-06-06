import logging
import logging.config
from os import stat
import fire
from typing import Dict, List
import pandas as pd
import numpy as np
import math
from sklearn.linear_model import LinearRegression
import alphashape

from ism_model import config as cfg
from ism_model.utils.dataset import Data


class Model():
    def __init__(self, L_0: float, Y_0: float):
        logging.info("Started the ISM model ...")

        self.L_0 = L_0
        self.Y_0 = Y_0

        self.sigma = 0.0
        self.delta = 0.0
        self.rho = 0.0

        self.a = 0.0
        self.gamma = 0.0
        self.alpha_k = 0.0
        self.n = 0.0
        self.s = 0.0
        self.delta = 0.0

    @staticmethod
    def create_params_set(coeffs: Dict[str, str], n_splits: int) -> pd.DataFrame:
        parsed_coeffs = {
            param: np.linspace(
                start=float(coeffs[param].split(";")[0]),
                stop=float(coeffs[param].split(";")[1]),
                num=n_splits
            ) for param in coeffs
        }
        grid_df = pd.DataFrame(
            data=np.vstack(np.meshgrid(*parsed_coeffs.values())).reshape(len(parsed_coeffs), -1).T,
            columns=parsed_coeffs.keys()
        )
        return grid_df

    @staticmethod
    def calc_gdp(Y_0: float, a: float, b: float, L: float, L_0: float, K: float, K_0: float) -> float:
        labour = (L/L_0)**(-b+1e-8)
        capital = (K/K_0)**(-b+1e-8)
        return Y_0*((a*labour + (1-a)*capital)**(-1/(b+1e-8)))

    @staticmethod
    def calc_capital(J: float, mu: float, K: float) -> float:
        return J + (1-mu)*K

    @staticmethod
    def calc_export(delta: float, Y: float, pi_e: float) -> float:
        return delta*Y/pi_e

    @staticmethod
    def calc_import(rho: float, delta: float, Y: float, pi_i: float) -> float:
        return rho*(1-delta)*Y/pi_i

    @staticmethod
    def calc_investments(sigma: float, rho: float, delta: float, Y: float, pi_j: float) -> float:
        return sigma*(1 + rho*(1 - delta))*Y/pi_j

    @staticmethod
    def calc_sigma(pi_j_list: List[float], J_list: List[float],
                   Y_list: List[float], pi_i_list: List[float], Imp_list: List[float]):
        sigma = [
            (pi_j*J)/(Y + pi_i*Imp) for pi_j, J, Y, pi_i, Imp in zip(pi_j_list, J_list, Y_list, pi_i_list, Imp_list)
        ]
        return sigma

    @staticmethod
    def calc_delta(pi_e_list: List[float], E_list: List[float], Y_list: List[float]):
        delta = [
            pi_e*E/Y for pi_e, E, Y in zip(pi_e_list, E_list, Y_list)
        ]
        return delta

    @staticmethod
    def calc_rho(pi_i_list: List[float], Imp_list: List[float],
                 Y_list: List[float], pi_e_list: List[float], E_list: List[float]):
        rho = [
            (pi_i*Imp)/(Y - pi_e*E) for pi_i, Imp, Y, pi_e, E in zip(pi_i_list, Imp_list, Y_list, pi_e_list, E_list)
        ]
        return rho

    @staticmethod
    def fit_coeffs(data: List[float]):
        LinReg = LinearRegression(n_jobs=-1)
        y = np.array(data)
        X = np.arange(len(data)).reshape(-1, 1)
        LinReg.fit(X, y)
        pred = LinReg.predict(X)
        return pred

    @staticmethod
    def fit_labour(self, data: List[float]):
        LinReg = LinearRegression(n_jobs=-1)
        y = np.log(np.array(data))
        X = np.arange(len(data)).reshape(-1, 1)
        LinReg.fit(X, y)
        return math.exp(LinReg.intercept_), LinReg.coef_[0]

    @staticmethod
    def fit_exponential(data: List[float]):
        LinReg = LinearRegression(n_jobs=-1)
        y = np.log(np.array(data))
        X = np.arange(len(data)).reshape(-1, 1)
        LinReg.fit(X, y)
        return math.exp(LinReg.intercept_), LinReg.coef_[0]

    @staticmethod
    def fit_polinominal(data: List[float], power: int):
        LinReg = LinearRegression(n_jobs=-1)
        y = np.array(data)
        X = np.array(
            [np.power(
                np.arange(len(data)),
                pow
            ) for pow in range(1, power+1)]
        ).T
        LinReg.fit(X, y)
        pred = LinReg.predict(X)
        return pred

    @staticmethod
    def create_shell(points: np.array, alpha: float):
        hull = alphashape.alphashape(points, alpha)
        hull_pts = hull.exterior.coords.xy
        return hull_pts

    @staticmethod
    def calc_pareto_front(points: np.array, return_mask: bool = True):
        is_efficient = np.arange(points.shape[0])
        n_points = points.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(points):
            nondominated_point_mask = np.any(points < points[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            points = points[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient

    def set_static_params(self, sigma: float, delta: float, rho: float) -> None:
        self.sigma = sigma
        self.delta = delta
        self.rho = rho

    def load_data(self, source_file: str):
        self.dataset = Data(source_file=source_file).load()
        self.dataset = self.dataset[[
            cfg.GDP_COL,
            cfg.LABOUR_COL
        ]]

    def save_data(self):
        pass

    def compute_metrics(self):
        pass

    def run(self):
        pass


if __name__ == "__main__":
    fire.Fire(Model)
