from src.data_loader import load_data
from src.model_builder import build_cnn
from src.trainer import train_model
from src.evaluator import evaluate_model

if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test, num_classes = load_data("data/DSL-StrongPasswordData.csv")
    input_shape = (X_train.shape[1], 1)

    # Build model
    model = build_cnn(input_shape, num_classes)

    # Train model
    history = train_model(model, X_train, y_train, X_test, y_test)

    # Evaluate
    evaluate_model(model, X_test, y_test)
