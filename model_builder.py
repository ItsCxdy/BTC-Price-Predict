import tensorflow as tf
from tensorflow.keras import layers, models

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    A single Transformer block.
    """
    # Attention Layer (The "Reasoning" part)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs # Skip Connection

    # Feed Forward Layer (The "Processing" part)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_cat2_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Stack multiple Transformer blocks
    # We use ~200k parameters logic (Lightweight GPT)
    x = transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)
    x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128, dropout=0.1)

    # Condense the 60 minutes of reasoning into a single vector
    x = layers.GlobalAveragePooling1D()(x)
    
    # Final decision layers
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    
    # Output: The Predicted Price (1 value)
    outputs = layers.Dense(1)(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name="CAT2_Transformer")
    
    # Compile with Adam optimizer and MSE loss (Standard for Regression)
    model.compile(optimizer="adam", loss="mse")
    
    return model