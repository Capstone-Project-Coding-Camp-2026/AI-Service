class ModelRegistry:
    def __init__(self):
        # Model 1: NLP Classification
        self.classify_model = None
        self.classify_metadata = None
        
        # Model 2: Forecasting
        self.forecast_model = None
        self.forecast_metadata = None
        
        # Model 3: What-If Lab
        self.whatif_model = None
        self.whatif_metadata = None

model_registry = ModelRegistry()