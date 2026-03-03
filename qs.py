import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests, adfuller, coint
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR
from linearmodels import PanelOLS, RandomEffects, PooledOLS
from linearmodels.panel import compare
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor, RANSACRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
from datetime import datetime
import os
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 100)



class DataPreprocessor:
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.df_raw = None
        self.df_panel = None
        
    def load_data(self):
        print("=" * 80)
        print("DATA LOADING AND PREPROCESSING")
        print("=" * 80)
        
        try:
            if self.file_path.endswith('.xlsx') or self.file_path.endswith('.xls'):
                self.df_raw = pd.read_excel(self.file_path, header=0)
            elif self.file_path.endswith('.csv'):
                try:
                    self.df_raw = pd.read_csv(self.file_path, header=0, encoding='utf-8')
                except:
                    self.df_raw = pd.read_csv(self.file_path, header=0, encoding='latin-1', sep=';')
            else:
                raise ValueError("Unsupported file format")
            
            print(f"✓ Data successfully loaded: {self.file_path}")
            print(f"✓ Dimensions: {self.df_raw.shape}")
            print(f"✓ Number of columns: {len(self.df_raw.columns)}")
            
            self.df_raw.columns = [str(col).strip().replace('\n', ' ') 
                                 for col in self.df_raw.columns]
            
            return True
            
        except Exception as e:
            print(f"✗ Error: {e}")
            return False
    
    def create_panel_data(self):
        print("\nCreating panel data format...")
        
        country_col = None
        for col in self.df_raw.columns:
            if any(keyword in str(col).lower() for keyword in ['country', 'ülke', 'ulke']):
                country_col = col
                break
        
        if country_col is None:
            country_col = self.df_raw.columns[0]
        
        panel_data = []
        
        for idx, row in self.df_raw.iterrows():
            country = row[country_col]
            if pd.isna(country):
                continue
            
            country = str(country).strip()
            
            for year in range(2015, 2027):
                year_str = str(year)
                
                univ_cols = [col for col in self.df_raw.columns 
                           if year_str in col and ('universit' in col.lower() or 'univ' in col.lower())]
                gdp_cols = [col for col in self.df_raw.columns 
                          if year_str in col and ('gdp' in col.lower() or 'gsyih' in col.lower())]
                
                if univ_cols and gdp_cols:
                    univ_val = row[univ_cols[0]]
                    gdp_val = row[gdp_cols[0]]
                    
                    if not (pd.isna(univ_val) or pd.isna(gdp_val)):
                        try:
                            if isinstance(gdp_val, str):
                                gdp_val = gdp_val.replace(',', '.')
                            if isinstance(univ_val, str):
                                univ_val = univ_val.replace(',', '.')
                            
                            panel_data.append({
                                'Country': country,
                                'Year': year,
                                'Universities': float(univ_val),
                                'GDP_Billion': float(gdp_val)
                            })
                        except:
                            continue
        
        self.df_panel = pd.DataFrame(panel_data)
        
        if len(self.df_panel) == 0:
            print("✗ Could not create panel data!")
            return False
        
        self.df_panel['ln_GDP'] = np.log(self.df_panel['GDP_Billion'] + 1)
        self.df_panel['ln_Universities'] = np.log(self.df_panel['Universities'] + 1)
        self.df_panel['GDP_per_Univ'] = self.df_panel['GDP_Billion'] / (self.df_panel['Universities'] + 1)
        self.df_panel['Year_Squared'] = self.df_panel['Year'] ** 2
        
        self.df_panel = self.df_panel.sort_values(['Country', 'Year']).reset_index(drop=True)
        
        print(f"✓ Panel data created:")
        print(f"  - Observations: {len(self.df_panel):,}")
        print(f"  - Countries: {self.df_panel['Country'].nunique()}")
        print(f"  - Year range: {self.df_panel['Year'].min()}-{self.df_panel['Year'].max()}")
        
        return True
    
    def add_missing_values_analysis(self):
        print("\nMissing value analysis:")
        missing = self.df_panel.isnull().sum()
        print(missing[missing > 0])
        
        if missing.sum() > 0:
            self.df_panel = self.df_panel.groupby('Country').apply(
                lambda x: x.ffill().bfill()
            ).reset_index(drop=True)
            print("✓ Missing values filled")
    
    def detect_outliers(self):
        print("\nOutlier analysis:")
        
        Q1 = self.df_panel[['Universities', 'GDP_Billion']].quantile(0.25)
        Q3 = self.df_panel[['Universities', 'GDP_Billion']].quantile(0.75)
        IQR = Q3 - Q1
        
        outliers = ((self.df_panel[['Universities', 'GDP_Billion']] < (Q1 - 1.5 * IQR)) | 
                   (self.df_panel[['Universities', 'GDP_Billion']] > (Q3 + 1.5 * IQR))).sum()
        
        print(f"Number of outliers:")
        print(f"  Universities: {outliers['Universities']}")
        print(f"  GDP_Billion: {outliers['GDP_Billion']}")
        
        return outliers
    
    def get_data(self):
        return self.df_panel.copy()



class AdvancedStatisticalAnalysis:
    
    def __init__(self, df):
        self.df = df
        self.results = {}
        
    def panel_unit_root_test(self):
        print("\n" + "=" * 80)
        print("PANEL UNIT ROOT TEST (Simplified)")
        print("=" * 80)
        
        countries = self.df['Country'].unique()
        stationary_counts = {'Universities': 0, 'GDP_Billion': 0}
        
        print("\nCountry-based stationarity test:")
        print("-" * 60)
        
        for country in countries[:10]:
            country_data = self.df[self.df['Country'] == country]
            
            if len(country_data) > 4:
                try:
                    result_uni = adfuller(country_data['Universities'].values)
                    result_gdp = adfuller(country_data['GDP_Billion'].values)
                    
                    uni_stationary = result_uni[1] < 0.05
                    gdp_stationary = result_gdp[1] < 0.05
                    
                    if uni_stationary:
                        stationary_counts['Universities'] += 1
                    if gdp_stationary:
                        stationary_counts['GDP_Billion'] += 1
                    
                    print(f"{country:20s}: Uni p={result_uni[1]:.3f} ({'Stationary' if uni_stationary else 'Non-stationary'}) | "
                          f"GDP p={result_gdp[1]:.3f} ({'Stationary' if gdp_stationary else 'Non-stationary'})")
                except:
                    continue
        
        print("\nResults:")
        print(f"Stationary university series: {stationary_counts['Universities']}/{min(10, len(countries))}")
        print(f"Stationary GDP series: {stationary_counts['GDP_Billion']}/{min(10, len(countries))}")
        
        self.results['stationarity'] = stationary_counts
        return stationary_counts
    
    def cointegration_test(self):
        print("\n" + "=" * 80)
        print("COINTEGRATION TEST")
        print("=" * 80)
        
        countries = self.df['Country'].unique()
        cointegrated_count = 0
        
        print("\nCountry-based cointegration test (Johansen/Engle-Granger type):")
        print("-" * 60)
        
        for country in countries[:10]:
            country_data = self.df[self.df['Country'] == country]
            
            if len(country_data) > 4:
                try:
                    X = sm.add_constant(country_data['ln_GDP'].values)
                    y = country_data['ln_Universities'].values
                    
                    model = sm.OLS(y, X).fit()
                    residuals = model.resid
                    
                    result = adfuller(residuals)
                    
                    if result[1] < 0.05:
                        cointegrated_count += 1
                        print(f"{country:20s}: p={result[1]:.3f} (Cointegrated ✓)")
                    else:
                        print(f"{country:20s}: p={result[1]:.3f} (Not cointegrated)")
                except:
                    print(f"{country:20s}: Test could not be performed")
        
        print(f"\nNumber of cointegrated countries: {cointegrated_count}/{min(10, len(countries))}")
        
        self.results['cointegration'] = cointegrated_count
        return cointegrated_count
    
    def advanced_correlation_analysis(self):
        print("\n" + "=" * 80)
        print("ADVANCED CORRELATION ANALYSIS")
        print("=" * 80)
        
        methods = ['pearson', 'spearman', 'kendall']
        corr_results = {}
        
        for method in methods:
            corr_matrix = self.df[['Universities', 'GDP_Billion', 'ln_GDP', 'ln_Universities']].corr(method=method)
            corr_results[method] = corr_matrix
            print(f"\n{method.upper()} Correlation Matrix:")
            print(corr_matrix.round(3))
        
        print("\nPARTIAL CORRELATION (Controlling for time effect):")
        try:
            self.df['Year_Centered'] = self.df['Year'] - self.df['Year'].mean()
            X = sm.add_constant(self.df[['Year_Centered']])
            
            model_uni = sm.OLS(self.df['ln_Universities'], X).fit()
            model_gdp = sm.OLS(self.df['ln_GDP'], X).fit()
            
            residuals_uni = model_uni.resid
            residuals_gdp = model_gdp.resid
            
            partial_corr = np.corrcoef(residuals_uni, residuals_gdp)[0, 1]
            print(f"Correlation controlled for time trend: {partial_corr:.3f}")
            
            corr_results['partial'] = partial_corr
            
        except Exception as e:
            print(f"Partial correlation error: {e}")
        
        self.results['correlation'] = corr_results
        return corr_results
    
    def granger_causality_analysis(self):
        print("\n" + "=" * 80)
        print("GRANGER CAUSALITY TEST")
        print("=" * 80)
        
        results_summary = {'GDP→Univ': [], 'Univ→GDP': []}
        
        countries = self.df['Country'].unique()[:5]
        
        for country in countries:
            print(f"\n{country}:")
            print("-" * 40)
            
            country_data = self.df[self.df['Country'] == country].sort_values('Year')
            
            if len(country_data) > 5:
                try:
                    data = country_data[['ln_Universities', 'ln_GDP']].values
                    
                    test_result = grangercausalitytests(data, maxlag=1, verbose=False)
                    
                    for lag, result in test_result.items():
                        p_value_1 = result[0]['ssr_ftest'][1]
                        p_value_2 = result[0]['ssr_chi2test'][1]
                        
                        if p_value_1 < 0.05:
                            results_summary['GDP→Univ'].append((country, p_value_1))
                            print(f"  Lag {lag}: GDP → Univ (p={p_value_1:.3f}) ✓")
                        else:
                            print(f"  Lag {lag}: GDP → Univ (p={p_value_1:.3f})")
                            
                        if p_value_2 < 0.05:
                            results_summary['Univ→GDP'].append((country, p_value_2))
                            print(f"  Lag {lag}: Univ → GDP (p={p_value_2:.3f}) ✓")
                        else:
                            print(f"  Lag {lag}: Univ → GDP (p={p_value_2:.3f})")
                            
                except Exception as e:
                    print(f"  Test could not be performed: {e}")
        
        print("\n" + "=" * 40)
        print("GRANGER TEST SUMMARY:")
        print(f"GDP → Univ causality: {len(results_summary['GDP→Univ'])}/{len(countries)} countries")
        print(f"Univ → GDP causality: {len(results_summary['Univ→GDP'])}/{len(countries)} countries")
        
        self.results['granger'] = results_summary
        return results_summary


class AdvancedRegressionModels:
    
    def __init__(self, df):
        self.df = df
        self.models = {}
        self.results = {}
        
    def run_dynamic_panel_gmm(self):
        print("\n" + "=" * 80)
        print("DYNAMIC PANEL MODEL ANALYSIS")
        print("=" * 80)
        
        df_dynamic = self.df.copy()
        df_dynamic['Univ_lag1'] = df_dynamic.groupby('Country')['ln_Universities'].shift(1)
        df_dynamic['GDP_lag1'] = df_dynamic.groupby('Country')['ln_GDP'].shift(1)
        
        df_dynamic = df_dynamic.dropna(subset=['Univ_lag1', 'GDP_lag1'])
        
        if len(df_dynamic) == 0:
            print("Insufficient data!")
            return None
        
        df_panel = df_dynamic.set_index(['Country', 'Year'])
        
        X = df_panel[['Univ_lag1', 'ln_GDP', 'GDP_lag1']]
        X = pd.concat([pd.Series(1, index=X.index, name='const'), X], axis=1)
        y = df_panel['ln_Universities']
        
        try:
            model = PanelOLS(y, X, entity_effects=True)
            results = model.fit(cov_type='clustered', cluster_entity=True)
            
            print("\nDYNAMIC PANEL MODEL RESULTS:")
            print(results.summary)
            
            print("\nINTERPRETATION:")
            if 'ln_GDP' in results.params:
                coef = results.params['ln_GDP']
                pval = results.pvalues['ln_GDP']
                
                print(f"GDP elasticity: {coef:.4f}")
                print(f"Short-term effect: 1% increase in GDP → {coef:.2f}% increase in Universities")
                
                if 'Univ_lag1' in results.params:
                    lag_coef = results.params['Univ_lag1']
                    print(f"Long-term effect: {coef/(1-lag_coef):.2f}% (dynamic adjustment)")
            
            self.results['dynamic_panel'] = results
            return results
            
        except Exception as e:
            print(f"Model error: {e}")
            return None
    
    def run_machine_learning_models(self):
        print("\n" + "=" * 80)
        print("MACHINE LEARNING MODELS")
        print("=" * 80)
        
        features = ['ln_GDP', 'Year']
        X = self.df[features].fillna(0)
        y = self.df['ln_Universities']
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Huber Regression': HuberRegressor(),
            'RANSAC Regression': RANSACRegressor(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n{name}:")
            print("-" * 30)
            
            try:
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, 
                                          scoring='r2', n_jobs=-1)
                
                print(f"CV R² scores: {cv_scores.round(3)}")
                print(f"Average R²: {cv_scores.mean():.3f} (±{cv_scores.std():.3f})")
                
                model.fit(X_scaled, y)
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    print("Feature importance:")
                    for feat, imp in zip(features, importances):
                        print(f"  {feat}: {imp:.3f}")
                
                results[name] = {
                    'model': model,
                    'cv_scores': cv_scores,
                    'mean_r2': cv_scores.mean()
                }
                
            except Exception as e:
                print(f"Error: {e}")
        
        if results:
            best_model_name = max(results.keys(), key=lambda x: results[x]['mean_r2'])
            print(f"\n✓ BEST MODEL: {best_model_name} (R² = {results[best_model_name]['mean_r2']:.3f})")
        
        self.results['ml_models'] = results
        return results
    
    def run_robust_panel_regression(self):
        print("\n" + "=" * 80)
        print("ROBUST PANEL REGRESSION ANALYSIS")
        print("=" * 80)
        
        df_panel = self.df.set_index(['Country', 'Year'])
        
        model_types = ['Fixed Effects', 'Random Effects', 'Pooled OLS']
        results_comparison = {}
        
        X = df_panel[['ln_GDP']]
        X = pd.concat([pd.Series(1, index=X.index, name='const'), X], axis=1)
        y = df_panel['ln_Universities']
        
        try:
            model_fe = PanelOLS(y, X, entity_effects=True)
            results_fe = model_fe.fit(cov_type='robust')
            results_comparison['Fixed Effects'] = results_fe
            print("\nFIXED EFFECTS (Robust):")
            print(f"  ln_GDP coefficient: {results_fe.params['ln_GDP']:.4f}")
            print(f"  p-value: {results_fe.pvalues['ln_GDP']:.4f}")
            print(f"  R²: {results_fe.rsquared:.3f}")
        except:
            print("Fixed Effects model could not be run")
        
        try:
            model_re = RandomEffects(y, X)
            results_re = model_re.fit(cov_type='robust')
            results_comparison['Random Effects'] = results_re
            print("\nRANDOM EFFECTS (Robust):")
            print(f"  ln_GDP coefficient: {results_re.params['ln_GDP']:.4f}")
            print(f"  p-value: {results_re.pvalues['ln_GDP']:.4f}")
            print(f"  R²: {results_re.rsquared:.3f}")
        except:
            print("Random Effects model could not be run")
        
        try:
            model_ols = PooledOLS(y, X)
            results_ols = model_ols.fit(cov_type='robust')
            results_comparison['Pooled OLS'] = results_ols
            print("\nPOOLED OLS (Robust):")
            print(f"  ln_GDP coefficient: {results_ols.params['ln_GDP']:.4f}")
            print(f"  p-value: {results_ols.pvalues['ln_GDP']:.4f}")
            print(f"  R²: {results_ols.rsquared:.3f}")
        except:
            print("Pooled OLS model could not be run")
        
        if 'Fixed Effects' in results_comparison and 'Random Effects' in results_comparison:
            try:
                print("\nHAUSMAN TEST:")
                hausman = compare(results_comparison['Fixed Effects'], 
                                results_comparison['Random Effects'])
                print(hausman.summary)
            except:
                print("Hausman test could not be performed")
        
        self.results['robust_panel'] = results_comparison
        return results_comparison


class TimeSeriesForecasting:
    
    def __init__(self, df):
        self.df = df
        
    def arima_forecasting(self):
        print("\n" + "=" * 80)
        print("ARIMA TIME SERIES FORECASTING")
        print("=" * 80)
        
        countries = self.df['Country'].unique()[:3]
        
        forecasts = {}
        
        for country in countries:
            print(f"\nARIMA forecast for {country}:")
            print("-" * 40)
            
            country_data = self.df[self.df['Country'] == country].sort_values('Year')
            
            if len(country_data) > 5:
                series = country_data['Universities'].values
                
                try:
                    model = ARIMA(series, order=(1, 1, 1))
                    model_fit = model.fit()
                    
                    forecast = model_fit.forecast(steps=5)
                    
                    print(f"  AIC: {model_fit.aic:.2f}")
                    print(f"  BIC: {model_fit.bic:.2f}")
                    print(f"  2026-2030 forecasts: {forecast.round(1)}")
                    
                    forecasts[country] = {
                        'model': model_fit,
                        'forecast': forecast,
                        'aic': model_fit.aic,
                        'bic': model_fit.bic
                    }
                    
                except Exception as e:
                    print(f"  ARIMA error: {e}")
        
        return forecasts
    
    def var_model_analysis(self):
        print("\n" + "=" * 80)
        print("VAR (VECTOR AUTOREGRESSION) MODEL")
        print("=" * 80)
        
        top_country = self.df.groupby('Country')['GDP_Billion'].mean().idxmax()
        country_data = self.df[self.df['Country'] == top_country].sort_values('Year')
        
        print(f"VAR model for {top_country}:")
        
        if len(country_data) > 5:
            data = country_data[['ln_Universities', 'ln_GDP']].copy()
            
            try:
                model = VAR(data)
                results = model.fit(maxlags=2, ic='aic')
                
                print("\nVAR Model Results:")
                print(results.summary())
                
                print("\nGranger Causality Test (within VAR):")
                granger_test = results.test_causality('ln_Universities', 'ln_GDP', kind='f')
                print(f"GDP → Univ: p = {granger_test.pvalue:.4f}")
                
                print("\nImpulse Response Analysis:")
                irf = results.irf(periods=5)
                
                forecast = results.forecast(data.values[-results.k_ar:], steps=5)
                print(f"\n5-Year Forecast (2026-2030):")
                print(f"  University: {np.exp(forecast[:, 0])}")
                print(f"  GDP: {np.exp(forecast[:, 1])}")
                
                return {
                    'results': results,
                    'granger_test': granger_test,
                    'forecast': forecast
                }
                
            except Exception as e:
                print(f"VAR model error: {e}")
        
        return None

class AdvancedVisualization:
    
    def __init__(self, df):
        self.df = df
        
    def create_3d_visualizations(self):
        print("\nCreating 3D visualizations...")
        
        fig = plt.figure(figsize=(18, 12))
        
        ax1 = fig.add_subplot(231, projection='3d')
        scatter1 = ax1.scatter(self.df['Year'], self.df['GDP_Billion'], 
                             self.df['Universities'], 
                             c=self.df['Year'], cmap='viridis', alpha=0.6)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('GDP (Billion $)')
        ax1.set_zlabel('Universities')
        ax1.set_title('3D: Year vs GDP vs Universities')
        
        ax2 = fig.add_subplot(232, projection='3d')
        
        country_avg = self.df.groupby('Country').agg({
            'GDP_Billion': 'mean',
            'Universities': 'mean'
        }).reset_index()
        
        top_countries = country_avg.nlargest(20, 'GDP_Billion')
        
        x_pos = np.arange(len(top_countries))
        y_pos = np.array([2020] * len(top_countries))
        z_pos = np.zeros(len(top_countries))
        
        dx = dy = np.ones(len(top_countries))
        dz_gdp = top_countries['GDP_Billion'].values / 100
        dz_uni = top_countries['Universities'].values
        
        ax2.bar3d(x_pos, y_pos, z_pos, dx, dy, dz_gdp, color='skyblue', alpha=0.8)
        ax2.set_xlabel('Countries')
        ax2.set_ylabel('Year')
        ax2.set_zlabel('Average GDP (Scaled)')
        ax2.set_title('Top 20 Countries: Average GDP (3D Bars)')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(top_countries['Country'], rotation=90, fontsize=8)
        
        ax3 = fig.add_subplot(233)
        top_10 = self.df.groupby('Country')['Universities'].mean().nlargest(10).index
        top_data = self.df[self.df['Country'].isin(top_10)]
        
        sns.violinplot(data=top_data, x='Country', y='Universities', ax=ax3)
        ax3.set_title('Top 10 Countries: Universities Distribution')
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha='right')
        
        ax4 = fig.add_subplot(234)
        corr_matrix = self.df[['Universities', 'GDP_Billion', 'ln_GDP', 'ln_Universities']].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   square=True, ax=ax4)
        ax4.set_title('Correlation Heatmap')
        
        ax5 = fig.add_subplot(235)
        
        top_5 = self.df.groupby('Country')['Universities'].mean().nlargest(5).index
        
        for country in top_5:
            country_data = self.df[self.df['Country'] == country].sort_values('Year')
            ax5.plot(country_data['Year'], country_data['Universities'], 
                    marker='o', linewidth=2, label=country)
        
        ax5.set_xlabel('Year')
        ax5.set_ylabel('Universities')
        ax5.set_title('Top 5 Countries: University Trends')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        ax6 = fig.add_subplot(236)
        
        scatter = ax6.scatter(self.df['GDP_Billion'], self.df['Universities'],
                            c=self.df['Year'], cmap='plasma', alpha=0.6)
        
        x = self.df['GDP_Billion'].values
        y = self.df['Universities'].values
        
        coeffs = np.polyfit(x, y, 2)
        poly = np.poly1d(coeffs)
        
        x_sorted = np.sort(x)
        ax6.plot(x_sorted, poly(x_sorted), 'r-', linewidth=2, 
                label=f'Poly fit (R²={np.corrcoef(y, poly(x))[0,1]:.2f})')
        
        ax6.set_xlabel('GDP (Billion $)')
        ax6.set_ylabel('Universities')
        ax6.set_title('GDP vs Universities with Polynomial Regression')
        ax6.legend()
        
        plt.colorbar(scatter, ax=ax6, label='Year')
        
        plt.tight_layout()
        plt.savefig('Advanced_Visualizations.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✓ Advanced visualizations created")
        
    def create_interactive_plots(self):
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            print("\nCreating interactive graphs...")
            
            fig1 = px.scatter(self.df, x='GDP_Billion', y='Universities',
                            color='Country', size='Universities',
                            hover_data=['Year', 'GDP_Billion', 'Universities'],
                            title='Interactive: GDP vs Universities by Country',
                            labels={'GDP_Billion': 'GDP (Billion $)',
                                  'Universities': 'QS Top 500 Universities'})
            
            fig1.write_html("Interactive_Scatter.html")
            print("✓ Interactive scatter plot saved")
            
            fig2 = px.scatter(self.df, x='GDP_Billion', y='Universities',
                            animation_frame='Year', animation_group='Country',
                            color='Country', hover_name='Country',
                            size='Universities',
                            title='Animated: GDP vs Universities Over Time',
                            labels={'GDP_Billion': 'GDP (Billion $)',
                                  'Universities': 'QS Top 500 Universities'})
            
            fig2.write_html("Animated_Time_Series.html")
            print("✓ Animated time series saved")
            
            return True
            
        except ImportError:
            print("Plotly library not installed. Install: pip install plotly")
            return False



class AdvancedForecastingSystem:
    
    def __init__(self, df):
        self.df = df
        
    def hybrid_forecasting_model(self):
        print("\n" + "=" * 80)
        print("HYBRID FORECASTING MODEL (2026-2030)")
        print("=" * 80)
        
        print("\n1. GDP PROJECTIONS:")
        
        projections = []
        countries = self.df['Country'].unique()
        
        for country in countries:
            country_data = self.df[self.df['Country'] == country]
            
            if len(country_data) < 3:
                continue
            
            recent = country_data[country_data['Year'] >= 2023]
            if len(recent) == 0:
                recent = country_data.tail(3)
            
            gdp_values = recent['GDP_Billion'].values
            if len(gdp_values) > 1:
                growth_rates = []
                for i in range(1, len(gdp_values)):
                    if gdp_values[i-1] > 0:
                        growth = (gdp_values[i] - gdp_values[i-1]) / gdp_values[i-1]
                        growth_rates.append(growth)
                
                if growth_rates:
                    avg_growth = np.mean(growth_rates)
                else:
                    avg_gdp = country_data['GDP_Billion'].mean()
                    if avg_gdp > 1000:
                        avg_growth = 0.025
                    elif avg_gdp > 100:
                        avg_growth = 0.035
                    else:
                        avg_growth = 0.045
            else:
                avg_growth = 0.03
            
            last_data = country_data[country_data['Year'] == 2025]
            if len(last_data) == 0:
                continue
            
            current_gdp = last_data['GDP_Billion'].values[0]
            current_univ = last_data['Universities'].values[0]
            
            for year in range(2026, 2031):
                years_ahead = year - 2025
                
                projected_gdp = current_gdp * ((1 + avg_growth) ** years_ahead)
                
                if len(country_data) > 2:
                    X = sm.add_constant(country_data['ln_GDP'])
                    y = country_data['ln_Universities']
                    
                    try:
                        model = sm.OLS(y, X).fit()
                        projected_ln_gdp = np.log(projected_gdp + 1)
                        projected_ln_univ = model.params['const'] + model.params['ln_GDP'] * projected_ln_gdp
                        projected_univ_reg = np.exp(projected_ln_univ)
                    except:
                        projected_univ_reg = current_univ * ((1 + avg_growth * 0.5) ** years_ahead)
                else:
                    projected_univ_reg = current_univ * ((1 + avg_growth * 0.5) ** years_ahead)
                
                if len(country_data) > 3:
                    years = country_data['Year'].values.reshape(-1, 1)
                    univs = country_data['Universities'].values
                    
                    trend_model = np.poly1d(np.polyfit(years.flatten(), univs, 1))
                    projected_univ_trend = trend_model(year)
                else:
                    projected_univ_trend = current_univ
                
                projected_univ = 0.6 * projected_univ_reg + 0.4 * projected_univ_trend
                
                projected_univ = max(0, min(projected_univ, 100))
                projected_univ = round(projected_univ, 1)
                
                projections.append({
                    'Country': country,
                    'Year': year,
                    'Projected_GDP_Billion': round(projected_gdp, 2),
                    'Projected_Universities': projected_univ,
                    'GDP_Growth_Rate_Pct': round(avg_growth * 100, 2),
                    'Projection_Method': 'Hybrid'
                })
        
        projections_df = pd.DataFrame(projections)
        
        print("\n2. PROJECTION ANALYSIS:")
        
        yearly_totals = projections_df.groupby('Year')['Projected_Universities'].sum()
        print(f"\nTotal University Count by Year:")
        for year, total in yearly_totals.items():
            print(f"  {year}: {total:.1f}")
        
        top_2030 = projections_df[projections_df['Year'] == 2030].nlargest(10, 'Projected_Universities')
        print(f"\nTop 10 Countries Expected to Have Highest University Count in 2030:")
        for i, (_, row) in enumerate(top_2030.iterrows(), 1):
            current_2025 = self.df[(self.df['Country'] == row['Country']) & 
                                 (self.df['Year'] == 2025)]['Universities']
            if len(current_2025) > 0:
                growth = ((row['Projected_Universities'] / current_2025.values[0]) - 1) * 100
                growth_str = f"(+{growth:.1f}%)"
            else:
                growth_str = ""
            
            print(f"  {i:2d}. {row['Country']:20s}: {row['Projected_Universities']:5.1f} {growth_str}")
        
        print(f"\nAverage Increase by GDP Group:")
        projections_df['GDP_Group'] = pd.cut(projections_df['Projected_GDP_Billion'],
                                          bins=[0, 100, 500, 1000, float('inf')],
                                          labels=['<100B', '100-500B', '500-1000B', '>1000B'])
        
        group_stats = projections_df[projections_df['Year'] == 2030].groupby('GDP_Group').agg({
            'Projected_Universities': ['mean', 'std', 'count']
        }).round(2)
        
        print(group_stats)
        
        projections_df.to_csv('Hybrid_Forecasts_2026_2030.csv', index=False, encoding='utf-8')
        print(f"\n✓ Hybrid forecasts saved to 'Hybrid_Forecasts_2026_2030.csv'")
        
        return projections_df


class ComprehensiveReportGenerator:
    
    def __init__(self, df, all_results):
        self.df = df
        self.results = all_results
        self.report_date = datetime.now().strftime("%d/%m/%Y %H:%M")
        

    
   
def main():
    print("\n" + "=" * 100)
    print("QS UNIVERSITY RANKING AND GDP RELATIONSHIP - ADVANCED ANALYSIS SYSTEM")
    print("=" * 100)
    print("Optimized Version with Best Methods")
    print("=" * 100)
    
    print("\n1. SYSTEM CHECK AND DEPENDENCIES")
    print("-" * 50)
    
    required_libs = ['pandas', 'numpy', 'statsmodels', 
                    'linearmodels', 'scipy', 'matplotlib', 'seaborn']
    
    missing_libs = []
    for lib in required_libs:
        try:
            __import__(lib)
            print(f"✓ {lib}")
        except ImportError:
            print(f"✗ {lib} - Missing")
            missing_libs.append(lib)
    
    if missing_libs:
        print(f"\nInstall missing libraries:")
        print(f"pip install {' '.join(missing_libs)}")
        return
    
    file_path = input("\nEnter data file path (press Enter to search for 'data.xlsx'): ").strip()
    
    if not file_path:
        for f in ['data.xlsx', 'data.csv', 'qs_data.xlsx', 'qs_data.csv']:
            if os.path.exists(f):
                file_path = f
                break
        
        if not file_path:
            print("✗ Data file not found!")
            return
    
    print(f"\n✓ File to be used: {file_path}")
    
    print("\n" + "=" * 100)
    print("2. DATA PREPROCESSING AND PANEL DATA CREATION")
    print("=" * 100)
    
    preprocessor = DataPreprocessor(file_path)
    if not preprocessor.load_data():
        return
    
    if not preprocessor.create_panel_data():
        return
    
    preprocessor.add_missing_values_analysis()
    preprocessor.detect_outliers()
    
    df = preprocessor.get_data()
    
    print("\n" + "=" * 100)
    print("3. ADVANCED STATISTICAL ANALYSIS")
    print("=" * 100)
    
    stat_analyzer = AdvancedStatisticalAnalysis(df)
    stat_results = {}
    
    stat_results['stationarity'] = stat_analyzer.panel_unit_root_test()
    stat_results['cointegration'] = stat_analyzer.cointegration_test()
    stat_results['correlation'] = stat_analyzer.advanced_correlation_analysis()
    stat_results['granger'] = stat_analyzer.granger_causality_analysis()
    
    print("\n" + "=" * 100)
    print("4. ADVANCED REGRESSION MODELS")
    print("=" * 100)
    
    reg_analyzer = AdvancedRegressionModels(df)
    reg_results = {}
    
    reg_results['dynamic_panel'] = reg_analyzer.run_dynamic_panel_gmm()
    reg_results['ml_models'] = reg_analyzer.run_machine_learning_models()
    reg_results['robust_panel'] = reg_analyzer.run_robust_panel_regression()
    
    print("\n" + "=" * 100)
    print("5. TIME SERIES FORECASTING MODELS")
    print("=" * 100)
    
    ts_forecaster = TimeSeriesForecasting(df)
    ts_results = {}
    
    ts_results['arima'] = ts_forecaster.arima_forecasting()
    ts_results['var'] = ts_forecaster.var_model_analysis()
    
    print("\n" + "=" * 100)
    print("6. ADVANCED VISUALIZATION")
    print("=" * 100)
    
    visualizer = AdvancedVisualization(df)
    visualizer.create_3d_visualizations()
    visualizer.create_interactive_plots()
    
    print("\n" + "=" * 100)
    print("7. HYBRID FORECASTING MODEL AND PROJECTIONS")
    print("=" * 100)
    
    forecaster = AdvancedForecastingSystem(df)
    projections = forecaster.hybrid_forecasting_model()
    
    print("\n" + "=" * 100)
    print("8. COMPREHENSIVE REPORT GENERATION")
    print("=" * 100)
    
    all_results = {
        **stat_results,
        **reg_results,
        **ts_results,
        'projections': projections
    }
    
    report_generator = ComprehensiveReportGenerator(df, all_results)
   
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETED SUCCESSFULLY! ")
    print("=" * 100)
    

    return {
        'data': df,
        'statistical_results': stat_results,
        'regression_results': reg_results,
        'time_series_results': ts_results,
        'projections': projections
    }

if __name__ == "__main__":
    try:
        results = main()
        if results:
            print("\n✓ All analyses completed successfully!")
    except KeyboardInterrupt:
        print("\n\n! Analysis stopped by user.")
    except Exception as e:
        print(f"\n\n! Unexpected error: {e}")
        import traceback
        traceback.print_exc()