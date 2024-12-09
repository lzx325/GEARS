{
    set -e
    module purge
	if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
		source "$HOME/anaconda3/etc/profile.d/conda.sh"
	else
		echo "conda not found"; exit 1;
	fi 
    conda deactivate
    conda activate GEARS
    # n_combo = 66 withGO {'mse': 0.007486132, 'mse_de': 0.14973044, 'pearson': 0.9933527078140116, 'pearson_de': 0.922683957572962}
    # n_combo = 99 withGO {'mse': 0.008115389, 'mse_de': 0.16270663, 'pearson': 0.992829593017479, 'pearson_de': 0.9067734413674233}
    # n_combo = 128 withGO {'mse': 0.0094368225, 'mse_de': 0.18870312, 'pearson': 0.9913421158903198, 'pearson_de': 0.900616878353798}
    # n_combo = 66 noGO {'mse': 0.0070986864, 'mse_de': 0.14471243, 'pearson': 0.9937114747133087, 'pearson_de': 0.9261356558674668}
    # n_combo = 99 noGO {'mse': 0.007606517, 'mse_de': 0.16081785, 'pearson': 0.9932533895608461, 'pearson_de': 0.912607218827436}
    # n_combo = 128 noGO {'mse': 0.008326985, 'mse_de': 0.17705853, 'pearson': 0.9926360464657371, 'pearson_de': 0.9043960708869316}
    python model_tutorial.py 66 test &
    python model_tutorial.py 99 test &
    python model_tutorial.py 128 test &
}