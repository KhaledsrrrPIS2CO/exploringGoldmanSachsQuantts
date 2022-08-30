import warnings
from datetime import date

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from gs_quant.datetime import business_day_offset
from gs_quant.markets import PricingContext, BackToTheFuturePricingContext
from gs_quant.risk import RollFwd, MarketDataPattern, MarketDataShock, MarketDataShockBasedScenario, MarketDataShockType
from gs_quant.instrument import FXOption, IRSwaption
from gs_quant.timeseries import *
from gs_quant.timeseries import percentiles

warnings.filterwarnings('ignore')
sns.set(style="darkgrid", color_codes=True)

from gs_quant.session import GsSession
# external users should substitute their client id and secret; please skip this step if using internal jupyterhub
GsSession.use(client_id=None, client_secret=None, scopes=('run_analytics',))

# 1: FX entry point vs richness¶
# Let's pull GS FX Spot and GS FX Implied Volatility and look at implied vs
# realized vol as well as current implied level as percentile relative to
# the last 2 years.

def format_df(data_dict):
    df = pd.concat(data_dict, axis=1)
    df.columns = data_dict.keys()
    return df.fillna(method='ffill').dropna()


g10 = ['USDJPY', 'EURUSD', 'AUDUSD', 'GBPUSD', 'USDCAD', 'USDNOK', 'NZDUSD', 'USDSEK', 'USDCHF', 'AUDJPY']
start_date = date(2005, 8, 26)
end_date = business_day_offset(date.today(), -1, roll='preceding')
fxspot_dataset, fxvol_dataset = Dataset('FXSPOT_PREMIUM'), Dataset('FXIMPLIEDVOL_PREMIUM')

spot_data, impvol_data, spot_fx = {}, {}, {}
for cross in g10:
    spot = fxspot_dataset.get_data(start_date, end_date, bbid=cross)[['spot']].drop_duplicates(keep='last')
    spot_fx[cross] = spot['spot']
    spot_data[cross] = volatility(spot['spot'], 63)  # realized vol
    vol = fxvol_dataset.get_data(start_date, end_date, bbid=cross, tenor='3m', deltaStrike='DN', location='NYC')[
        ['impliedVolatility']]
    impvol_data[cross] = vol.drop_duplicates(keep='last') * 100

spdata, ivdata = format_df(spot_data), format_df(impvol_data)
diff = ivdata.subtract(spdata).dropna()

_slice = ivdata['2018-09-01': '2020-09-08']
pct_rank = {}
for x in _slice.columns:
    pct = percentiles(_slice[x])
    pct_rank[x] = pct.iloc[-1]

for fx in pct_rank:
    plt.scatter(pct_rank[fx], diff[fx]['2020-09-08'])
    plt.legend(pct_rank.keys(), loc='best', bbox_to_anchor=(0.9, -0.13), ncol=3)

plt.xlabel('Percentile of Current Implied Vol')
plt.ylabel('Implied vs Realized Vol')
plt.title('Entry Point vs Richness')
plt.show()

# 2: Downside sensitivity to SPX¶
# Let's now look at beta and correlation with SPX across G10.

spx_spot = Dataset('TREOD').get_data(start_date, end_date, bbid='SPX')[['closePrice']]
spx_spot = spx_spot.fillna(method='ffill').dropna()
df = pd.DataFrame(spx_spot)

# FX Spot data
fx_spots = format_df(spot_fx)
data = pd.concat([spx_spot, fx_spots], axis=1).dropna()
data.columns = ['SPX'] + g10

beta_spx, corr_spx = {}, {}

# calculate rolling 84d or 4m beta to S&P
for cross in g10:
    beta_spx[cross] = beta(data[cross],data['SPX'], 84)
    corr_spx[cross] = correlation(data['SPX'], data[cross], 84)

fig, axs = plt.subplots(5, 2, figsize=(18, 20))
for j in range(2):
    for i in range(5):
        color='tab:blue'
        axs[i,j].plot(beta_spx[g10[i + j*5]], color=color)
        axs[i,j].set_title(g10[i + j*5])
        color='tab:blue'
        axs[i,j].set_ylabel('Beta', color=color)
        axs[i,j].plot(beta_spx[g10[i + j*5]], color=color)
        ax2 = axs[i,j].twinx()
        color = 'tab:orange'
        ax2.plot(corr_spx[g10[i + j*5]], color=color)
        ax2.set_ylabel('Correlation', color=color)
plt.show()