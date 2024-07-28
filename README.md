# Macaque_PythonRuns
> Note: Model and Code taken from MACBSE repository: [https://github.com/ajoshiusc/macbse](https://github.com/ajoshiusc/macbse)

### Data Sources
Datasets used from PRIMatE Data Exchange([link](https://fcon_1000.projects.nitrc.org/indi/indiPRIME.html)):
-   Aix-Marseille Université: [link](https://fcon_1000.projects.nitrc.org/indi/PRIME/amu.html)
-   University of Minnesota: [link](https://fcon_1000.projects.nitrc.org/indi/PRIME/uminn.html)
-   NIN Primate Brain Bank/Utrecht University: [link](https://fcon_1000.projects.nitrc.org/indi/PRIME/NINPBBUtrecht.html)

### How to run code
Store the input file in the same folder where `model.pth` and `macbse.py` are stored. Then run the code in the following format:
```
python macbse.py -i input.nii.gz -m model.pth -o output.bse.nii.gz
```
If you want the mask also, here's the format:
```
python macbse.py -i input.nii.gz -m model.pth -o output.bse.nii.gz --mask mask.nii.gz
```
