from main import main
import matplotlib.pyplot as plt

def plot_predictions(actual, predictions):
    """
    Memvisualisasikan hasil prediksi
    """
    plt.figure(figsize=(12,6))
    plt.plot(actual, label='Actual Price', color='blue')
    plt.plot(predictions, label='Predicted Price', color='red')
    plt.title('Bitcoin Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig('prediction_results.png')
    plt.show()

if __name__ == "__main__":
    print("Memulai proses prediksi cryptocurrency...")
    print("Loading data dan mempersiapkan model...")
    
    try:
        actual, predictions = main()
        
        print("\nPrediksi selesai!")
        print("\nMembuat visualisasi...")
        plot_predictions(actual, predictions)
        
        print("\nProses selesai! Hasil visualisasi telah disimpan sebagai 'prediction_results.png'")
        
    except Exception as e:
        print(f"\nTerjadi error: {str(e)}") 