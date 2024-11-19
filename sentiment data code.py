import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

# Step 1: Load Ticker-to-Sector Mapping and Standardize 'ticker'
# Read the CSV with tab delimiter and specify 'Ticker' as object dtype
ticker_sector_df = pd.read_csv('ticker_sector.csv', delimiter='\t', dtype={'Ticker': 'object'})

# Rename 'Ticker' to 'ticker' for consistency
ticker_sector_df.rename(columns={'Ticker': 'ticker'}, inplace=True)

# Standardize 'ticker' column: uppercase and strip whitespace
ticker_sector_df['ticker'] = ticker_sector_df['ticker'].str.upper().str.strip()

# Ensure 'Sector' is of type 'object'
ticker_sector_df['Sector'] = ticker_sector_df['Sector'].astype('object')

# Remove duplicate tickers, keeping the first occurrence
ticker_sector_df = ticker_sector_df.drop_duplicates(subset='ticker', keep='first')

# Convert to Dask DataFrame
ticker_sector_ddf = dd.from_pandas(ticker_sector_df, npartitions=1)

# Enforce 'object' dtype on 'ticker' in Dask DataFrame
ticker_sector_ddf['ticker'] = ticker_sector_ddf['ticker'].astype('object')

# Step 2: Load Parquet Files Using Dask
columns_needed = ['ticker', 'sent_merged']
df = dd.read_parquet('df*.parquet', columns=columns_needed)

# Standardize 'ticker' in df: uppercase and strip whitespace
df['ticker'] = df['ticker'].str.upper().str.strip()

# Ensure 'ticker' is of type 'object'
df['ticker'] = df['ticker'].astype('object')

# Ensure 'sent_merged' is of type float32
df['sent_merged'] = df['sent_merged'].astype('float32')

# Step 3: Enforce 'object' dtype on both 'ticker' columns before merging
df['ticker'] = df['ticker'].astype('object')
ticker_sector_ddf['ticker'] = ticker_sector_ddf['ticker'].astype('object')

# Step 4: Merge with Sector Information
df = df.merge(
    ticker_sector_ddf[['ticker', 'Sector']],  # Select only necessary columns
    on='ticker',
    how='left'
)

# Step 5: Handle Missing Sectors by Dropping Them
df = df.dropna(subset=['Sector'])

# Optionally, exclude sectors that are 'Unknown', 'Defunct', or 'ETF' if desired
excluded_sectors = ['Unknown', 'Defunct', 'ETF']
df = df[~df['Sector'].isin(excluded_sectors)]

# Ensure 'Sector' is categorical for efficient grouping
df['Sector'] = df['Sector'].astype('category')

# Persist DataFrame in memory if possible to optimize performance
df = df.persist()

# Step 6: Compute Sentiment Data by Sector with observed=True
with ProgressBar():
    sentiment_by_sector = df.groupby('Sector', observed=True)['sent_merged'].mean().compute(scheduler='single-threaded')

print("Sentiment by Sector:")
print(sentiment_by_sector)
