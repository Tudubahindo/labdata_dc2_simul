# labdata_dc2_simul
**Codice simulazioni**

il file simul.py contiene il codice di python usato per le simulazioni numeriche e anche per le altre analisi sui dati reali (e relativi plot). Per dovizia ho caricato anche i dataset puliti.

la funzione ``hopkins_statistic (X)`` restituisce, dato un dataframe di una colonna, la relativa statistica di Hopkins. L'ho copiata da [qui](https://github.com/prathmachowksey/Hopkins-Statistic-Clustering-Tendency/blob/master/.ipynb_checkpoints/Hopkins-Statistic-Clustering-Tendency-checkpoint.ipynb), quindi per dettagli sull'implementazione consultare la pagina originale. Il resto l'ho effettivamente scritto io.

la funzione ``hopkins_calibration (N, rho)``, chiamata così per il suo uso nella calibrazione di Hopkins ma in realtà di utilizzo più generale, restituisce un dateset simulato di N numeri tra 0 e 1, generato secondo il parametro rho:

1. Se rho == 0, il dataset è generato con distribuzione uniforme ``np.random.uniform (0,1,N)``.
2. Se rho > 0, il dataset è generato con distribuzione normale centrata in 0.5 e di larghezza 1/rho ``np.random.normal (0.5,1/rho,N)``.
3. Se rho < 0, il dataset è generato dividendo l'intervallo [0,1] in N pivot equidistanziati, e generando un punto per ogni pivot con una distribuzione normale centrata nel pivot e larga 1/(N x |rho|) ``np.random.normal (1/2N + i/N, 1/(N*np.abs(rho)), 1)``.

la funzione ``dataset_visualization ()`` genera tre dataset con ``hopkins_calibration (100, rho)``, con rho rispettivamente 0, 10 e -10, e li plotta con dei barcode.

la funzione ``sogliole (num, simulnum, rho, printing)``, chiamata così perché usata nella calibrazione delle sogliole ma in realtà di utilizzo più generale, genera ``simulnum`` dataset usando ``hopkins_calibration (num, rho)`` e di ogni dataset calcola la statistica di Hopkins *h* usando ``hopkins_statistic (X)``. La funzione restituisce quindi il valore medio di *h* tra tutti i dataset. Se poi si specifica ``printng==True`` la funzione stampa il valore massimo degli *h*, oltre alla media e al valor minimo, e il 2.5 e il 97.5 percentile, per la calibrazione delle soglie. Lo stesso per la deviazione standard relativa.

Nel ``main ()`` segue poi l'analisi sui dati reali.
