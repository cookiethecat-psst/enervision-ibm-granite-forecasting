"""
Comprehensive Test Suite for EnerVision TTM Models
Testing IBM Granite TTM integration and performance
"""

import unittest
import asyncio
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.enervision import (
    EnerVisionTTM,
    TTMForecaster,
    IBMEnsembleFusion,
    IndianBuildingContextAdapter,
    ForecastResult
)
from src.enervision.models.ttm_models import (
    create_ttm_model,
    UltraLightTTM,
    GraniteTimeMixer,
    EnsembleTTM
)
from src.enervision.models.evaluation import (
    TTMEvaluator,
    PerformanceBenchmark,
    EvaluationMetrics
)
from src.enervision.models.anomaly_detection import (
    TSPulseDetector,
    EnsembleAnomalyDetector,
    AnomalyResult
)


class TestTTMForecaster(unittest.TestCase):
    """Test TTM Forecaster functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.forecaster = TTMForecaster('512-96')
        self.sample_data = np.random.randn(512) * 10 + 100
        self.context_features = np.random.randn(512, 6)
        
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.forecaster.model)
        self.assertIsNotNone(self.forecaster.config)
        self.assertEqual(self.forecaster.config['context_length'], 512)
        self.assertEqual(self.forecaster.config['prediction_length'], 96)
        
    def test_forecast_generation(self):
        """Test forecast generation"""
        forecast = self.forecaster.forecast(
            historical_data=self.sample_data,
            exogenous_variables=self.context_features,
            prediction_length=96
        )
        
        self.assertIsNotNone(forecast)
        self.assertEqual(len(forecast), 96)
        self.assertTrue(np.all(np.isfinite(forecast)))
        
    def test_different_context_lengths(self):
        """Test different context length models"""
        models = ['512-96', '1024-96', '1536-96']
        
        for model_size in models:
            forecaster = TTMForecaster(model_size)
            context_len = int(model_size.split('-')[0])
            
            self.assertEqual(forecaster.config['context_length'], context_len)
            
            # Test with appropriate data size
            data = np.random.randn(context_len) * 10 + 100
            forecast = forecaster.forecast(data, prediction_length=24)
            
            self.assertEqual(len(forecast), 24)
    
    def test_edge_cases(self):
        """Test edge cases"""
        # Test with minimal data
        minimal_data = np.array([100.0])
        forecast = self.forecaster.forecast(minimal_data, prediction_length=10)
        self.assertEqual(len(forecast), 10)
        
        # Test with very long context
        long_data = np.random.randn(2000) * 10 + 100
        forecast = self.forecaster.forecast(long_data, prediction_length=96)
        self.assertEqual(len(forecast), 96)


class TestEnerVisionTTM(unittest.TestCase):
    """Test main EnerVision TTM system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.enervision = EnerVisionTTM()
        
        # Sample building data
        self.building_data = {
            'consumption': np.random.randn(512) * 10 + 100,
            'contextual_features': np.random.randn(512, 6),
            'location': 'bangalore',
            'category': 'it_campus'
        }
    
    def test_initialization(self):
        """Test system initialization"""
        self.assertIsNotNone(self.enervision.ttm_ensemble)
        self.assertEqual(len(self.enervision.ttm_ensemble), 3)
        self.assertIsNotNone(self.enervision.fusion_layer)
    
    @patch('src.enervision.EnerVisionTTM._watsonx_api_forecast')
    async def test_forecast_energy_load(self, mock_watsonx):
        """Test energy load forecasting"""
        # Mock watsonx API response
        mock_watsonx.return_value = np.random.randn(96) * 10 + 100
        
        result = await self.enervision.forecast_energy_load(
            self.building_data,
            horizon='24h'
        )
        
        self.assertIsInstance(result, ForecastResult)
        self.assertEqual(len(result.point_forecast), 96)
        self.assertIsNotNone(result.prediction_intervals)
        self.assertIsNotNone(result.anomaly_scores)
        self.assertIsNotNone(result.optimization_opportunities)
    
    def test_horizon_conversion(self):
        """Test horizon string conversion"""
        test_cases = {
            '15m': 1,
            '1h': 4,
            '24h': 96,
            '72h': 288
        }
        
        for horizon, expected_steps in test_cases.items():
            steps = self.enervision._convert_horizon_to_steps(horizon)
            self.assertEqual(steps, expected_steps)
    
    def test_savings_identification(self):
        """Test savings identification"""
        # Create mock forecast result
        forecast_result = ForecastResult(
            point_forecast=np.random.randn(96) * 10 + 100,
            prediction_intervals={'95': (np.zeros(96), np.ones(96))},
            anomaly_scores=np.random.rand(96),
            optimization_opportunities=[],
            metadata={}
        )
        
        savings = self.enervision.identify_savings(forecast_result)
        
        self.assertIn('total_potential_savings', savings)
        self.assertIn('peak_reduction', savings)
        self.assertIn('load_shifting', savings)
        self.assertIn('renewable_integration', savings)
        self.assertIn('annual_savings_inr', savings)


class TestIndianBuildingContextAdapter(unittest.TestCase):
    """Test Indian building context adaptation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.adapter = IndianBuildingContextAdapter()
        
        self.building_dataset = {
            'consumption': np.random.randn(1000) * 10 + 100,
            'location': 'mumbai',
            'category': 'commercial_office',
            'dates': pd.date_range('2025-01-01', periods=1000, freq='H').tolist()
        }
    
    def test_climate_zone_encoding(self):
        """Test climate zone encoding"""
        test_cases = {
            'delhi': 0,  # hot_dry
            'mumbai': 1,  # hot_humid
            'bangalore': 2,  # temperate
            'shimla': 3  # cold
        }
        
        for location, expected_code in test_cases.items():
            code = self.adapter._encode_indian_climate_zones(location)
            self.assertEqual(code, expected_code)
    
    def test_building_type_encoding(self):
        """Test building type encoding"""
        test_cases = {
            'it campus': 0,
            'commercial office': 1,
            'mixed use': 2,
            'educational': 3
        }
        
        for category, expected_code in test_cases.items():
            code = self.adapter._encode_indian_building_types(category)
            self.assertEqual(code, expected_code)
    
    def test_seasonal_extraction(self):
        """Test seasonal pattern extraction"""
        seasonality = self.adapter._extract_indian_seasonality(self.building_dataset)
        
        self.assertIsNotNone(seasonality)
        self.assertEqual(len(seasonality), 3)  # 3 factors
        
    def test_festival_encoding(self):
        """Test festival encoding"""
        festivals = self.adapter._encode_indian_festivals(self.building_dataset['dates'])
        
        self.assertIsNotNone(festivals)
        self.assertEqual(len(festivals), len(self.adapter.festivals))
    
    async def test_fine_tuning(self):
        """Test fine-tuning for Indian buildings"""
        model = TTMForecaster('512-96')
        
        adapted_model = await self.adapter.fine_tune_for_indian_buildings(
            self.building_dataset,
            model
        )
        
        self.assertIsNotNone(adapted_model)
        self.assertIn('indian_context', adapted_model.config)


class TestEnsembleFusion(unittest.TestCase):
    """Test ensemble fusion layer"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.fusion = IBMEnsembleFusion()
        
    def test_prediction_combination(self):
        """Test combining multiple predictions"""
        predictions = {
            'watsonx_api': {
                'forecast': np.random.randn(96) * 10 + 100
            },
            'local_ttm': np.random.randn(96) * 10 + 100
        }
        
        result = self.fusion.combine_predictions(predictions)
        
        self.assertIsInstance(result, ForecastResult)
        self.assertEqual(len(result.point_forecast), 96)
        self.assertIn('95', result.prediction_intervals)
        self.assertIsNotNone(result.anomaly_scores)
    
    def test_weight_normalization(self):
        """Test weight normalization"""
        total_weight = sum(self.fusion.weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=5)
    
    def test_anomaly_calculation(self):
        """Test anomaly score calculation"""
        forecast = np.random.randn(96) * 10 + 100
        anomaly_scores = self.fusion._calculate_anomaly_scores(forecast)
        
        self.assertEqual(len(anomaly_scores), len(forecast))
        self.assertTrue(np.all(anomaly_scores >= 0))


class TestTTMModels(unittest.TestCase):
    """Test TTM model implementations"""
    
    def test_ultralight_ttm(self):
        """Test ultra-lightweight TTM model"""
        model = UltraLightTTM(
            context_length=96,
            prediction_length=24,
            n_channels=7
        )
        
        # Test forward pass
        batch_size = 4
        x = torch.randn(batch_size, 96, 7)
        output = model(x)
        
        self.assertEqual(output.shape, (batch_size, 24, 7))
    
    def test_granite_time_mixer(self):
        """Test Granite Time Mixer model"""
        model = GraniteTimeMixer(
            context_length=512,
            prediction_length=96,
            n_channels=7
        )
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 7, 512)
        output = model(x)
        
        self.assertEqual(output.shape, (batch_size, 7, 96))
    
    def test_model_factory(self):
        """Test model factory function"""
        model_types = ['standard', 'ultralight', 'anomaly']
        
        for model_type in model_types:
            if model_type == 'ensemble':
                continue  # Skip ensemble for this test
                
            model = create_ttm_model(
                model_type,
                context_length=96,
                prediction_length=24,
                n_channels=7
            )
            
            self.assertIsNotNone(model)


class TestAnomalyDetection(unittest.TestCase):
    """Test anomaly detection functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.detector = TSPulseDetector()
        self.data = np.random.randn(1000) * 10 + 100
        
        # Add some anomalies
        self.data[100:105] = 200  # High anomaly
        self.data[500:503] = 20   # Low anomaly
    
    def test_tspulse_detection(self):
        """Test TSPulse anomaly detection"""
        self.detector.fit(self.data[:800])
        result = self.detector.detect(self.data[800:])
        
        self.assertIsInstance(result, AnomalyResult)
        self.assertEqual(len(result.scores), 200)
        self.assertEqual(len(result.anomalies), 200)
        self.assertIsNotNone(result.statistics)
    
    def test_ensemble_detector(self):
        """Test ensemble anomaly detector"""
        ensemble = EnsembleAnomalyDetector()
        ensemble.fit(self.data[:800])
        
        result = ensemble.detect(self.data[800:])
        
        self.assertIsInstance(result, AnomalyResult)
        self.assertIn('detector_agreement', result.statistics)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Test performance benchmarking"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.enervision = EnerVisionTTM()
        self.benchmark = PerformanceBenchmark(self.enervision)
    
    def test_latency_benchmark(self):
        """Test latency benchmarking"""
        results = self.benchmark.run_latency_benchmark(
            batch_sizes=[1, 10],
            context_lengths=[96, 512],
            num_runs=10
        )
        
        self.assertIsNotNone(results)
        self.assertIn('batch_1_context_96', results)
        
        for key, metrics in results.items():
            self.assertIn('mean_latency_ms', metrics)
            self.assertIn('p99_latency_ms', metrics)
            # Check if meets target
            self.assertLess(metrics['mean_latency_ms'], 100)
    
    def test_memory_benchmark(self):
        """Test memory benchmarking"""
        results = self.benchmark.run_memory_benchmark()
        
        self.assertIn('peak_memory_mb', results)
        self.assertIn('model_size_mb', results)
        
        # Check model size constraint
        self.assertLess(results['model_size_mb'], 10)  # Should be < 10MB
    
    def test_scalability_benchmark(self):
        """Test scalability benchmarking"""
        results = self.benchmark.run_scalability_benchmark()
        
        self.assertIn('100_buildings', results)
        
        for key, metrics in results.items():
            self.assertIn('throughput_buildings_per_second', metrics)
            self.assertIn('average_latency_ms', metrics)


class TestIntegration(unittest.TestCase):
    """Integration tests for complete workflow"""
    
    async def test_end_to_end_forecasting(self):
        """Test complete forecasting workflow"""
        # Initialize system
        enervision = EnerVisionTTM()
        
        # Generate sample data
        df = pd.DataFrame({
            'timestamp': pd.date_range('2025-01-01', periods=1000, freq='H'),
            'energy_consumption': np.random.randn(1000) * 10 + 100,
            'temperature': np.random.randn(1000) * 5 + 25,
            'humidity': np.random.randn(1000) * 10 + 60,
            'solar_irradiance': np.random.rand(1000) * 800,
            'occupancy_rate': np.random.rand(1000)
        })
        
        # Prepare building data
        building_data = {
            'consumption': df['energy_consumption'].values[:512],
            'contextual_features': df[['temperature', 'humidity', 'solar_irradiance', 
                                      'occupancy_rate']].values[:512],
            'location': 'bangalore',
            'category': 'it_campus'
        }
        
        # Generate forecast
        result = await enervision.forecast_energy_load(building_data, horizon='24h')
        
        # Validate results
        self.assertIsInstance(result, ForecastResult)
        self.assertEqual(len(result.point_forecast), 96)
        
        # Check accuracy (should be > 95%)
        # In real scenario, compare with actual values
        self.assertIsNotNone(result.metadata)
        
        # Check optimization opportunities
        self.assertTrue(len(result.optimization_opportunities) > 0)
        
        # Calculate savings
        savings = enervision.identify_savings(result)
        self.assertIn('total_potential_savings', savings)
        
        # Parse savings percentage
        savings_pct = float(savings['total_potential_savings'].rstrip('%'))
        self.assertGreater(savings_pct, 20)  # Should achieve > 20% savings


def run_async_test(coro):
    """Helper to run async tests"""
    loop = asyncio.get_event_loop()
    return loop.run_until_complete(coro)


if __name__ == '__main__':
    unittest.main()