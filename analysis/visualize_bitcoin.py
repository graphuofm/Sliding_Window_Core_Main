import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Read timeline data
df = pd.read_csv('bitcoin_timeline.txt', sep='\t')
df['Date'] = pd.to_datetime(df['Date'])

# Read anomaly data
anomalies = pd.read_csv('bitcoin_anomalies.txt', sep='\t')
anomalies['Date'] = pd.to_datetime(anomalies['Date'])

# Plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

# Max core over time
ax1.plot(df['Date'], df['MaxCore'], 'b-', linewidth=1)
ax1.set_ylabel('Maximum k-Core', fontsize=12)
ax1.set_title('Bitcoin Transaction Network Core Evolution (2009-2016)', fontsize=14)
ax1.grid(True, alpha=0.3)

# Mark anomalies
for _, anomaly in anomalies.iterrows():
    ax1.axvline(x=anomaly['Date'], color='red', alpha=0.5, linestyle='--')
    ax1.text(anomaly['Date'], ax1.get_ylim()[1]*0.9, anomaly['Event'], 
             rotation=90, verticalalignment='top', fontsize=8)

# Processing time
ax2.plot(df['Date'], df['ProcessingTime(ms)'], 'g-', linewidth=1)
ax2.set_ylabel('UCR Processing Time (ms)', fontsize=12)
ax2.set_xlabel('Date', fontsize=12)
ax2.grid(True, alpha=0.3)

# Format x-axis
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('bitcoin_core_evolution.png', dpi=300)
plt.show()
