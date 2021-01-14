# satellite_data
sat_data.py is a python wrapper to extract data from the ESA's DISCOS database (https://discosweb.esoc.esa.int/).
sat_plots.py is a plotting and analysis module which interacts with the data returned by sat_data.py.

Ensure that the python file is in your current directory or $PYTHONPATH. 
You will also need register for your own API token from the DISCOS website, and edit the file to include it. Alternatively, save the token in a plain .txt file in the project directory with filename ".esa_token.txt"

Please note: Extracting data from DISCOS for the first time will take a considerable amount of time for some databases (e.g. > 10 minutes). This data will be saved locally in the project directory so that it can be retreived quickly for future use. 

Usage example:

```python
> import sat_data                                                                                               
> df = sat_data.get_data(database = 'objects')                                                                  
> df[['DiscosID','IntlDes','SatName','ObjectType','Shape','Mass']].head()                                   
 
   DiscosID    IntlDes                  SatName   ObjectType       Shape     Mass
0         1  1957-001A  Sputnik (8K71PS) Blok-A  Rocket Body         Cyl  3964.32
1         2  1957-001B                Sputnik 1      Payload      Sphere    82.85
2         3  1957-002A                Sputnik 2      Payload  Cone + Cyl   503.77
3         4  1958-001A               Explorer 1      Payload         Cyl    13.88
4         5  1958-002B               Vanguard 1      Payload      Sphere     1.46

> import sat_plots
> p = sat_plots.SatPlots()
                                                                                  
Read data from file ./esa_data/fragmentations_2021-01-03.csv
Read data from file ./esa_data/fragmentation-event-types_2020-12-07.csv
Read data from file ./esa_data/objects_2021-01-13.csv
Read data from file ./esa_data/launches_2020-12-04.csv
Read data from file ./esa_data/reentries_2021-01-03.csv
Read data from file ./esa_data/launch-sites_2021-01-03.csv
Read data from file ./esa_data/launch-systems_2020-12-07.csv
Read data from file ./esa_data/launch-vehicles_2020-12-07.csv
Read data from file ./esa_data/propellants_2020-12-07.csv
Read data from file ./esa_data/entities_2021-01-03.csv
Read data from file ./esa_data/initial-orbits_2021-01-03.csv
Read data from file ./esa_data/destination-orbits_2021-01-03.csv
UCS data file found, generated on 2021-01-03
File: ./esa_data/ucsdata_2021-01-03.csv

> p.plot_pop_evolution(vars = ['Payload','Launches','Junk'], scale = 'linear') 

![alt text](https://github.com/jamesgrimmett/satellite_data/blob/main/plots/pop_growth_example.png?raw=true)
```
