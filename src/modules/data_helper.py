from markets_insights.core.environment import Environment
from datetime import date, timedelta
import pandas as pd
import os

import datetime
from markets_insights.datareader.data_reader import BhavCopyReader, DataReader, DateRangeCriteria, NseIndicesReader
from markets_insights.dataprocess.data_processor import HistoricalDataProcessor, MultiDataCalculationPipelines, CalculationPipelineBuilder, HistoricalDataProcessOptions
from markets_insights.calculations.base import DatePartsCalculationWorker
from markets_insights.dataprocess import data_processor
from markets_insights.calculations.base import BaseColumns

from modules.calc_workers import (
    create_volume_calc_pipeline, create_tech_flags_pipeline, create_liquidity_calc_pipeline,
    create_price_features_pipeline, create_target_cacl_pipeline, create_growth_calc_pipeline
)

cache_folder_path = '../cache-data'

default_includes: dict = {
    'indexes': [{
        'identifier': 'Nifty 50',
        'with_calc': True
    }, {
        'identifier': 'India VIX',
        'with_calc': False
    }]
}

# Setup environment
Environment.setup(cache_data_base_path=cache_folder_path)

def normalize_data(data: pd.DataFrame):
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def fetch_data(histDataProcessor: HistoricalDataProcessor, reader: DataReader,start_date: date, end_date: date):
    print(f"Fetching stock data from {start_date} to {end_date}\n")
    # Fetch data for date range
    result = histDataProcessor.process(reader, DateRangeCriteria(start_date, end_date))

    # Get daily data
    df = result.get_daily_data()

    print(f"Successfully fetched {len(df)} records")
    print(f"Columns: {list(df.columns)}")

    # Display sample
    print(f"\nData Summary:")
    if len(df) > 0:
        print(f"Date range in data: {df.iloc[:, 0]} to {df.iloc[-1, 0]}")
        print(f"\nFirst 5 records:")
        print(f"\nRecords: {histDataProcessor.dataset.get_daily_data().shape[0]}")
        print(df.head(5))
    else:
        print("No data was fetched")

def get_processed_data(histDataProcessor: HistoricalDataProcessor, 
        reader: DataReader, 
        data_identifier: str, 
        start_date: date, 
        end_date: date, 
        force_refresh=False, 
        run_calc: bool = True,
        include_target: bool = True
    ):
    file_path = os.path.join(cache_folder_path, "ML", f"{data_identifier}-{reader.name}_{start_date}-{end_date}.csv")
    if os.path.exists(file_path) and not force_refresh:
        print(f"Loading data from cache: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} records from cache")
    else:
        fetch_data(histDataProcessor, reader, start_date, end_date)
        
        if run_calc:
            # prepare calculation pipeline
            periods = [x for x in range(1, 16)]

            pipelines = data_processor.MultiDataCalculationPipelines()
            pipelines.set_item('forward_looking_fall', data_processor.CalculationPipelineBuilder.create_forward_looking_price_fall_pipeline(periods))
            pipelines.set_item('forward_looking_rise', data_processor.CalculationPipelineBuilder.create_forward_looking_price_rise_pipeline(periods))
            pipelines.set_item('rsi', data_processor.CalculationPipelineBuilder.create_rsi_calculation_pipeline(crossing_above_flag_value = 75, crossing_below_flag_value = 30, window = 14))
            pipelines.set_item('stoch_rsi', data_processor.CalculationPipelineBuilder.create_stoch_rsi_calculation_pipeline(crossing_above_flag_value = 80, crossing_below_flag_value = 20, window = 14))
            pipelines.set_item('sma', data_processor.CalculationPipelineBuilder.create_sma_calculation_pipeline([50, 100, 200]))
            pipelines.set_item('bbands', data_processor.CalculationPipelineBuilder.create_bb_calculation_pipeline(windows=[200], deviations=[2, 3]))
            pipelines.set_item('tech_flags', create_tech_flags_pipeline())
            pipelines.set_item('volume', create_volume_calc_pipeline(col=BaseColumns.Volume))
            pipelines.set_item('turnover', create_volume_calc_pipeline(col=BaseColumns.Turnover))
            pipelines.set_item('liquidity', create_liquidity_calc_pipeline(col=BaseColumns.Turnover))
            pipelines.set_item('price', create_price_features_pipeline())
            pipelines.set_item('growth', create_growth_calc_pipeline())
            pipelines.set_item('target', create_target_cacl_pipeline(target_value=5, stop_loss_value=2.5))
            histDataProcessor.set_calculation_pipelines(pipelines=pipelines)
            # run the pipeline and show results
            histDataProcessor.run_calculation_pipelines()

        df = histDataProcessor.dataset.get_daily_data()
        df.to_csv(file_path, index=False)
        print(f"Successfully processed and saved {len(df)} records to cache")
    df = normalize_data(df)
    return df
    
def load_data(data_identifier: str, start_date: date, end_date: date, force_refresh=False, includes: dict = {
        "Indexes": {}
    }):
    file_path = os.path.join(cache_folder_path, "ML", f"{data_identifier}_{start_date}-{end_date}.csv")
    if os.path.exists(file_path) and not force_refresh:
        print(f"Loading data from cache: {file_path}")
        df = pd.read_csv(file_path)
        print(f"Successfully loaded {len(df)} records from cache")
        return df
    # Create processor
    options = HistoricalDataProcessOptions(include_monthly_data=False, include_annual_data=False)
    histDataProcessor = HistoricalDataProcessor(options)
    data = get_processed_data(histDataProcessor, BhavCopyReader(), data_identifier, start_date, end_date)
    
    if default_includes and 'indexes' in default_includes:
        index_data = get_processed_data(histDataProcessor, NseIndicesReader(), f"{data_identifier}_index", 
                                        start_date, end_date, force_refresh=force_refresh, run_calc=True, include_target=False)
        for index in default_includes['indexes']:
            identifier: str = index['identifier']
            # Merge Index data with main data
            data = pd.merge(data, index_data[index_data['Identifier'] == identifier], on='Date', how='left', suffixes=('', f'__{identifier}'))
        data.to_csv(file_path, index=False)
    return data