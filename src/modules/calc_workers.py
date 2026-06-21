import numpy as np
import pandas as pd
from markets_insights.calculations.base import (
    CalculationWorker, BaseColumns,
    ColumnValueAboveFlagWorker, ColumnValueBelowFlagWorker,
    ColumnValueAboveAnotherColumnValueFlagWorker,
    ColumnChangeOverNDaysCalculationWorker
)
from markets_insights.core.core import Instrumentation
from markets_insights.dataprocess.data_processor import CalculationPipeline

class ColumnSmaCalculationWorker(CalculationWorker):
    def __init__(self, value_column: str, time_window: int = 20):
        super().__init__(value_column=value_column, time_window=int(time_window))
        self._columns.append(f'{value_column}Sma{time_window}')

    @Instrumentation.trace(name='ColumnSmaCalculationWorker')
    def add_calculated_columns(self, data: pd.DataFrame):
        grouped = data.groupby(self.get_group_cols(data.columns))
        data[self._columns[0]] = grouped[self._params['value_column']].transform(
            lambda x: x.rolling(self._params['time_window']).mean()
        )

class ColumnRelativeToSmaCalculationWorker(CalculationWorker):
    def __init__(self, value_column: str, time_window: int = 20):
        super().__init__(value_column=value_column, time_window=int(time_window))
        self._columns.append(f'Relative{value_column}{time_window}')

    @Instrumentation.trace(name='ColumnRelativeToSmaCalculationWorker')
    def add_calculated_columns(self, data: pd.DataFrame):
        sma_col = f'{self._params["value_column"]}Sma{self._params["time_window"]}'
        data[self._columns[0]] = data[self._params['value_column']] / data[sma_col]

class ColumnValueAboveScaledColumnValueFlagWorker(CalculationWorker):
    def __init__(self, value_column: str, reference_column: str, multiplier: float = 2.0):
        super().__init__(value_column=value_column, reference_column=reference_column, multiplier=float(multiplier))
        multiplier_str = str(int(multiplier)) if multiplier == int(multiplier) else str(multiplier)
        self._columns.append(f'{value_column}Above{multiplier_str}x{reference_column}')

    @Instrumentation.trace(name='ColumnValueAboveScaledColumnValueFlagWorker')
    def add_calculated_columns(self, data: pd.DataFrame):
        data[self._columns[0]] = (
            data[self._params['value_column']] > self._params['multiplier'] * data[self._params['reference_column']]
        ).astype(int)

class ColumnRisingWithVolumeConfirmationFlagWorker(CalculationWorker):
    def __init__(self, price_column: str, prev_price_column: str, volume_column: str, volume_threshold: float = 1.5):
        super().__init__(
            price_column=price_column, prev_price_column=prev_price_column,
            volume_column=volume_column, volume_threshold=float(volume_threshold)
        )
        self._columns.append(f'{price_column}VolumeConfirmation')

    @Instrumentation.trace(name='ColumnRisingWithVolumeConfirmationFlagWorker')
    def add_calculated_columns(self, data: pd.DataFrame):
        data[self._columns[0]] = (
            (data[self._params['price_column']] > data[self._params['prev_price_column']]) &
            (data[self._params['volume_column']] > self._params['volume_threshold'])
        ).astype(int)

class ColumnPercentileRankByDateCalculationWorker(CalculationWorker):
    def __init__(self, value_column: str):
        super().__init__(value_column=value_column)
        self._columns.append(f'{value_column}PercentileRank')

    @Instrumentation.trace(name='ColumnPercentileRankByDateCalculationWorker')
    def add_calculated_columns(self, data: pd.DataFrame):
        data[self._columns[0]] = data.groupby(BaseColumns.Date)[self._params['value_column']].rank(pct=True)

def create_volume_calc_pipeline(col:str, time_window: int = 20):
    pipeline = CalculationPipeline()
    pipeline.add_calculation_worker(ColumnSmaCalculationWorker(value_column=col, time_window=time_window))
    pipeline.add_calculation_worker(ColumnRelativeToSmaCalculationWorker(value_column=col, time_window=time_window))
    pipeline.add_calculation_worker(ColumnValueAboveScaledColumnValueFlagWorker(
        value_column=col,
        reference_column=f'{col}Sma{time_window}',
    ))
    pipeline.add_calculation_worker(ColumnRisingWithVolumeConfirmationFlagWorker(
        price_column=BaseColumns.Close,
        prev_price_column=BaseColumns.PreviousClose,
        volume_column=f'Relative{col}{time_window}',
    ))
    return pipeline

def create_liquidity_calc_pipeline(col: str = BaseColumns.Turnover, time_window: int = 20):
    pipeline = CalculationPipeline()
    pipeline.add_calculation_worker(ColumnPercentileRankByDateCalculationWorker(value_column=f'{col}Sma{time_window}'))
    return pipeline

def create_tech_flags_pipeline():
    pipeline = CalculationPipeline()
    # RSI threshold flags → RsiBelow30, RsiAbove75
    pipeline.add_calculation_worker(ColumnValueBelowFlagWorker('Rsi', 30))
    pipeline.add_calculation_worker(ColumnValueAboveFlagWorker('Rsi', 75))
    # Stoch RSI threshold flags → StochRsi_KBelow20, StochRsi_KAbove80
    pipeline.add_calculation_worker(ColumnValueBelowFlagWorker('StochRsi_K', 20))
    pipeline.add_calculation_worker(ColumnValueAboveFlagWorker('StochRsi_K', 80))
    # Price vs SMA flags → CloseAboveSma50, CloseAboveSma100, CloseAboveSma200
    pipeline.add_calculation_worker(ColumnValueAboveAnotherColumnValueFlagWorker('Close', 'Sma50'))
    pipeline.add_calculation_worker(ColumnValueAboveAnotherColumnValueFlagWorker('Close', 'Sma100'))
    pipeline.add_calculation_worker(ColumnValueAboveAnotherColumnValueFlagWorker('Close', 'Sma200'))
    return pipeline

class DailyRangePercCalculationWorker(CalculationWorker):
    def __init__(self):
        super().__init__()
        self._columns.append('DailyRangePerc')

    @Instrumentation.trace(name='DailyRangePercCalculationWorker')
    def add_calculated_columns(self, data: pd.DataFrame):
        data[self._columns[0]] = (
            (data[BaseColumns.High] - data[BaseColumns.Low]) / data[BaseColumns.Close]
        )

class GapPercCalculationWorker(CalculationWorker):
    def __init__(self):
        super().__init__()
        self._columns.append('GapPerc')

    @Instrumentation.trace(name='GapPercCalculationWorker')
    def add_calculated_columns(self, data: pd.DataFrame):
        data[self._columns[0]] = (
            (data[BaseColumns.Open] - data[BaseColumns.PreviousClose]) / data[BaseColumns.PreviousClose]
        )

class StopHitDayCalculationWorker(CalculationWorker):
    def __init__(self, stop_loss_value: float = 2.5):
        super().__init__(stop_loss_value=float(stop_loss_value))
        self._columns.append('StopHitDay')

    @Instrumentation.trace(name='StopHitDayCalculationWorker')
    def add_calculated_columns(self, data: pd.DataFrame):
        low_cols = [f'TroughPercInNext{i}Sessions' for i in range(1, 16)]
        mask = data[low_cols] >= self._params['stop_loss_value']
        result = mask.idxmax(axis=1).str.extract(r'(\d+)').astype(float).squeeze()
        result[~mask.any(axis=1)] = np.nan
        data[self._columns[0]] = result


class TargetHitDayCalculationWorker(CalculationWorker):
    def __init__(self, target_value: float = 5):
        super().__init__(target_value=float(target_value))
        self._columns.append('TargetHitDay')

    @Instrumentation.trace(name='TargetHitDayCalculationWorker')
    def add_calculated_columns(self, data: pd.DataFrame):
        high_cols = [f'PeakPercInNext{i}Sessions' for i in range(1, 16)]
        mask = data[high_cols] >= self._params['target_value']
        result = mask.idxmax(axis=1).str.extract(r'(\d+)').astype(float).squeeze()
        result[~mask.any(axis=1)] = np.nan
        data[self._columns[0]] = result


# class TargetCalculationWorker(CalculationWorker):
#     def __init__(self):
#         super().__init__()
#         self._columns.append('Target')

#     @Instrumentation.trace(name='TargetCalculationWorker')
#     def add_calculated_columns(self, data: pd.DataFrame):
#         target_hit_no_stop = (
#             data['TargetHitDay'].notna() &
#             data['StopHitDay'].isna()
#         )
#         target_hits_first = (
#             data['TargetHitDay'].notna() &
#             data['StopHitDay'].notna() &
#             (data['TargetHitDay'] < data['StopHitDay'])
#         )
#         data[self._columns[0]] = (target_hit_no_stop | target_hits_first).astype(int)


def create_price_features_pipeline():
    pipeline = CalculationPipeline()
    pipeline.add_calculation_worker(DailyRangePercCalculationWorker())
    pipeline.add_calculation_worker(GapPercCalculationWorker())
    return pipeline

def create_target_cacl_pipeline(target_value: float = 5, stop_loss_value: float = 2.5):
    pipeline = CalculationPipeline()
    pipeline.add_calculation_worker(TargetHitDayCalculationWorker(target_value=target_value))
    pipeline.add_calculation_worker(StopHitDayCalculationWorker(stop_loss_value=stop_loss_value))
    # pipeline.add_calculation_worker(TargetCalculationWorker())
    return pipeline

def create_growth_calc_pipeline(periods: list[int] = [20]):
    pipeline = CalculationPipeline()
    pipeline.add_calculation_worker(ColumnChangeOverNDaysCalculationWorker(value_column=BaseColumns.Close, N = periods[0]))
    # pipeline.add_calculation_worker(TargetCalculationWorker())
    return pipeline