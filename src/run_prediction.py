from main import main
import matplotlib.pyplot as plt
import seaborn as sns

def plot_analysis(actual, predictions, price_analysis, metrics):
    """
    Memvisualisasikan hasil prediksi dan analisis
    """
    # Set style
    plt.style.use('seaborn')
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Price Prediction
    ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
    ax1.plot(actual, label='Harga Aktual', color='blue', linewidth=2)
    ax1.plot(predictions, label='Prediksi', color='red', linewidth=2, linestyle='--')
    ax1.set_title(f'Prediksi Harga Bitcoin (MAPE: {metrics["mape"]:.2f}%)', fontsize=12, pad=15)
    ax1.set_xlabel('Periode')
    ax1.set_ylabel('Harga (USD)')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Price Status Gauge
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    score = price_analysis['score']
    colors = ['green' if score < 50 else 'red' if score > 50 else 'gray']
    ax2.barh(['Score'], [score], color=colors)
    ax2.set_xlim(0, 100)
    ax2.set_title(f"Status Harga: {price_analysis['status']}", fontsize=12)
    ax2.axvline(x=50, color='gray', linestyle='--')
    
    # Plot 3: RSI Indicator
    ax3 = plt.subplot2grid((2, 2), (1, 1))
    rsi = price_analysis['rsi']
    ax3.barh(['RSI'], [rsi], color='blue')
    ax3.set_xlim(0, 100)
    ax3.axvline(x=30, color='green', linestyle='--')
    ax3.axvline(x=70, color='red', linestyle='--')
    ax3.set_title('RSI Indicator', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('analysis_results.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Memulai analisis cryptocurrency...")
    
    try:
        # Terima 4 nilai return dari main()
        actual, predictions, price_analysis, metrics = main()
        
        if actual is not None and price_analysis is not None:
            print("\n=== Analisis Harga Cryptocurrency ===")
            print(f"Status: {price_analysis['status']}")
            print(f"Skor: {price_analysis['score']:.2f}/100")
            print("\nSignal yang terdeteksi:")
            for signal in price_analysis['signals']:
                print(f"- {signal}")
            print(f"\nHarga saat ini: ${price_analysis['current_price']:,.2f}")
            print(f"RSI: {price_analysis['rsi']:.2f}")
            print(f"Posisi dalam Bollinger Bands: {price_analysis['bb_position']:.2f}%")
            
            print("\nMembuat visualisasi...")
            plot_analysis(actual, predictions, price_analysis, metrics)
            print("\nAnalisis selesai! Hasil visualisasi telah disimpan sebagai 'analysis_results.png'")
            
    except Exception as e:
        print(f"\nTerjadi error dalam program utama:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}") 