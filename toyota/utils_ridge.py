import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Optional, List, Union
from sklearn.linear_model import Ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.graphics.gofplots import ProbPlot

def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, engine="python", encoding="utf8")

def get_metrics(y_test: pd.Series, y_pred: pd.Series) -> dict:
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2_validation = r2_score(y_test, y_pred)
    return {"mse": mse, "mae": mae, "rmse": rmse, "r2_validation": r2_validation}

style_talk = 'seaborn-talk'    #refer to plt.style.available

class SklearnRidgeDiagnostic:
    """
    Diagnostic plots to identify potential problems in a scikit-learn Ridge regression fit.
    Accepts a fitted Ridge model, X (features), and y (target).
    """

    def __init__(
        self,
        model: Ridge,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        feature_names: Optional[List[str]] = None
    ) -> None:
        """
        Args:
            model: Fitted sklearn Ridge instance.
            X: Features used for fitting/prediction (n_samples, n_features).
            y: True target values (n_samples,).
            feature_names: Optional list of feature names.
        """
        if not hasattr(model, "coef_"):
            raise ValueError("Model must be a fitted sklearn Ridge instance.")

        self.model = model
        self.X = X.values if isinstance(X, pd.DataFrame) else X
        self.y_true = y.values if isinstance(y, (pd.Series, pd.DataFrame)) else y
        self.y_predict = model.predict(self.X)
        self.n_samples, self.n_features = self.X.shape

        if feature_names is not None:
            self.xvar_names = feature_names
        elif hasattr(X, "columns"):
            self.xvar_names = list(X.columns)
        else:
            self.xvar_names = [f"x{i}" for i in range(self.n_features)]

        # Residuals
        self.residual = self.y_true - self.y_predict

        # Standardized residuals
        self.residual_norm = self._standardized_residuals()

        # Leverage (hat values)
        self.leverage = self._leverage()

        # Cook's distance
        self.cooks_distance = self._cooks_distance()

        self.nparams = self.n_features
        self.nresids = self.n_samples

        # Properties to store outlier indices for each plot
        self.residuals_vs_fitted_outliers = None
        self.qq_plot_outliers = None
        self.scale_location_outliers = None
        self.leverage_plot_outliers = None

    def _standardized_residuals(self):
        # Standardized residuals: (residual / estimated std of residuals)
        residual_std = np.std(self.residual, ddof=self.n_features)
        return self.residual / (residual_std + 1e-8)

    def _leverage(self):
        # Hat matrix diagonal: h_ii = x_i^T (X^T X)^-1 x_i
        X = self.X
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        try:
            XtX_inv = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X.T @ X)
        hat_diag = np.sum(X @ XtX_inv * X, axis=1)
        return hat_diag

    def _cooks_distance(self):
        # Cook's distance: D_i = (standardized_residual^2 * h_ii) / [p * (1 - h_ii)^2]
        res_norm = self.residual_norm
        h = self.leverage
        p = self.n_features
        cooks = (res_norm ** 2) * h / (p * (1 - h) ** 2 + 1e-8)
        return cooks

    def __call__(self, plot_context='seaborn-v0_8-paper', **kwargs):
        if plot_context not in plt.style.available:
            plot_context = 'default'
        with plt.style.context(plot_context):
            fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,10))
            self.predictted_real(ax=ax[0,0])
            self.histogram(ax=ax[0,1])
            self.residual_plot(ax=ax[0,2])
            self.qq_plot(ax=ax[1,0])
            self.scale_location_plot(ax=ax[1,1])
            self.leverage_plot(
                ax=ax[1,2],
                high_leverage_threshold=kwargs.get('high_leverage_threshold'),
                cooks_threshold=kwargs.get('cooks_threshold'))
            fig.savefig("./images/residual_plots.png")
        return fig, ax

    def residual_plot(self, ax=None, show_outliers=True):
        """
        Residual vs Fitted Plot
        If show_outliers is True, stores the indices of the top 3 outliers in self.residuals_vs_fitted_outliers.
        """
        if ax is None:
            fig, ax = plt.subplots()

        sns.residplot(
            x=self.y_predict,
            y=self.residual,
            lowess=True,
            scatter_kws={'alpha': 0.5},
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        residual_abs = np.abs(self.residual)
        abs_resid = np.flip(np.argsort(residual_abs), 0)
        abs_resid_top_3 = abs_resid[:3]
        outliers = []
        for i in abs_resid_top_3:
            outliers.append(i)
            ax.annotate(
                str(i),
                xy=(self.y_predict[i], self.residual[i]),
                color='C3')

        if show_outliers:
            self.residuals_vs_fitted_outliers = outliers
        else:
            self.residuals_vs_fitted_outliers = None

        ax.set_title('Residuals vs Fitted', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel('Residuals')
        return ax

    def qq_plot(self, ax=None, show_outliers=True):
        """
        Standardized Residual vs Theoretical Quantile plot
        If show_outliers is True, stores the indices of the top 3 outliers in self.qq_plot_outliers.
        """
        if ax is None:
            fig, ax = plt.subplots()

        QQ = ProbPlot(self.residual_norm)
        fig = QQ.qqplot(line='45', alpha=0.5, lw=1, ax=ax)

        # annotations
        abs_norm_resid = np.flip(np.argsort(np.abs(self.residual_norm)), 0)
        abs_norm_resid_top_3 = abs_norm_resid[:3]
        outliers = []
        for i, x, y in self.__qq_top_resid(QQ.theoretical_quantiles, abs_norm_resid_top_3):
            outliers.append(i)
            ax.annotate(
                str(i),
                xy=(x, y),
                ha='right',
                color='C3')

        if show_outliers:
            self.qq_plot_outliers = outliers
        else:
            self.qq_plot_outliers = None

        ax.set_title('Normal Q-Q', fontweight="bold")
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Standardized Residuals')
        return ax

    def scale_location_plot(self, ax=None, show_outliers=True):
        """
        Sqrt(Standardized Residual) vs Fitted values plot
        If show_outliers is True, stores the indices of the top 3 outliers in self.scale_location_outliers.
        """
        if ax is None:
            fig, ax = plt.subplots()

        residual_norm_abs_sqrt = np.sqrt(np.abs(self.residual_norm))

        ax.scatter(self.y_predict, residual_norm_abs_sqrt, alpha=0.5)
        sns.regplot(
            x=self.y_predict,
            y=residual_norm_abs_sqrt,
            scatter=False, ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        abs_sq_norm_resid = np.flip(np.argsort(residual_norm_abs_sqrt), 0)
        abs_sq_norm_resid_top_3 = abs_sq_norm_resid[:3]
        outliers = []
        for i in abs_sq_norm_resid_top_3:
            outliers.append(i)
            ax.annotate(
                str(i),
                xy=(self.y_predict[i], residual_norm_abs_sqrt[i]),
                color='C3')

        if show_outliers:
            self.scale_location_outliers = outliers
        else:
            self.scale_location_outliers = None

        ax.set_title('Scale-Location', fontweight="bold")
        ax.set_xlabel('Fitted values')
        ax.set_ylabel(r'$\sqrt{|\mathrm{Standardized\ Residuals}|}$')
        return ax

    def leverage_plot(self, ax=None, high_leverage_threshold=False, cooks_threshold='baseR', show_outliers=True):
        """
        Residual vs Leverage plot
        If show_outliers is True, stores the indices of the top 3 outliers in self.leverage_plot_outliers.
        """
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(
            self.leverage,
            self.residual_norm,
            alpha=0.5)

        sns.regplot(
            x=self.leverage,
            y=self.residual_norm,
            scatter=False,
            ci=False,
            lowess=True,
            line_kws={'color': 'red', 'lw': 1, 'alpha': 0.8},
            ax=ax)

        # annotations
        leverage_top_3 = np.flip(np.argsort(self.cooks_distance), 0)[:3]
        outliers = []
        for i in leverage_top_3:
            outliers.append(i)
            ax.annotate(
                str(i),
                xy=(self.leverage[i], self.residual_norm[i]),
                color='C3')

        if show_outliers:
            self.leverage_plot_outliers = outliers
        else:
            self.leverage_plot_outliers = None

        factors = []
        if cooks_threshold == 'baseR' or cooks_threshold is None:
            factors = [1, 0.5]
        elif cooks_threshold == 'convention':
            factors = [4/self.nresids]
        elif cooks_threshold == 'dof':
            factors = [4/ (self.nresids - self.nparams)]
        else:
            raise ValueError("threshold_method must be one of the following: 'convention', 'dof', or 'baseR' (default)")
        for i, factor in enumerate(factors):
            label = "Cook's distance" if i == 0 else None
            xtemp, ytemp = self.__cooks_dist_line(factor)
            ax.plot(xtemp, ytemp, label=label, lw=1.25, ls='--', color='red')
            ax.plot(xtemp, np.negative(ytemp), lw=1.25, ls='--', color='red')

        if high_leverage_threshold:
            high_leverage = 2 * self.nparams / self.nresids
            if max(self.leverage) > high_leverage:
                ax.axvline(high_leverage, label='High leverage', ls='-.', color='purple', lw=1)

        ax.axhline(0, ls='dotted', color='black', lw=1.25)
        ax.set_xlim(0, max(self.leverage)+0.01)
        ax.set_ylim(min(self.residual_norm)-0.1, max(self.residual_norm)+0.1)
        ax.set_title('Residuals vs Leverage', fontweight="bold")
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Standardized Residuals')
        plt.legend(loc='best')
        return ax

    def histogram(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        sns.histplot(self.residual_norm, bins=30, kde=True, ax=ax)
        ax.axvline(0, color='red', linestyle='--')
        ax.set_title("Histogram of residuals", fontweight="bold")
        ax.set_xlabel('Standardized residuals')
        return ax

    def predictted_real(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        ax.scatter(x=self.y_true, y=self.y_predict)
        ax.plot([min(self.y_true), max(self.y_true)], [min(self.y_true), max(self.y_true)], color='red', linestyle='--')
        ax.set_xlim(0, max(self.y_true) + 2)
        ax.set_ylim(0, max(self.y_predict) + 2)
        ax.set_title("Predicted vs Real", fontweight="bold")
        ax.set_ylabel("Predicted values")
        ax.set_xlabel("Real values")
        return ax

    def vif_table(self):
        """
        VIF table

        VIF, the variance inflation factor, is a measure of multicollinearity.
        VIF > 5 for a variable indicates that it is highly collinear with the
        other input variables.
        """
        vif_df = pd.DataFrame()
        vif_df["Features"] = self.xvar_names
        vif_df["VIF Factor"] = [variance_inflation_factor(self.X, i) for i in range(self.X.shape[1])]

        return (vif_df
                .sort_values("VIF Factor")
                .round(2))

    def __cooks_dist_line(self, factor):
        """
        Helper function for plotting Cook's distance curves
        """
        p = self.nparams
        formula = lambda x: np.sqrt((factor * p * (1 - x)) / x)
        x = np.linspace(0.001, max(self.leverage), 50)
        y = formula(x)
        return x, y

    def __qq_top_resid(self, quantiles, top_residual_indices):
        """
        Helper generator function yielding the index and coordinates
        """
        offset = 0
        quant_index = 0
        previous_is_negative = None
        for resid_index in top_residual_indices:
            y = self.residual_norm[resid_index]
            is_negative = y < 0
            if previous_is_negative is None or previous_is_negative == is_negative:
                offset += 1
            else:
                quant_index -= offset
            x = quantiles[quant_index] if is_negative else np.flip(quantiles, 0)[quant_index]
            quant_index += 1
            previous_is_negative = is_negative
            yield resid_index, x, y