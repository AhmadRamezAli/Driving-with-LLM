from pydantic import BaseModel, Field

class Prediction(BaseModel):
    """Represents the prediction output from the model."""

    caption: str = Field(..., description="A text caption describing the predicted action.")
    accelerate: float = Field(..., description="The predicted acceleration value.")
    brake: float = Field(..., description="The predicted brake value.")
    steering: float = Field(..., description="The predicted steering value.") 
    time_taken: float = Field(..., description="The time taken to predict the action.")
