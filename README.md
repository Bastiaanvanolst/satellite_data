# satellite_data
A python wrapper to extract data from the ESA's DISCOS database (https://discosweb.esoc.esa.int/). Soon to come: plotting functions.
Ensure that the python file is in your current directory or $PYTHONPATH. 
You will also need register for your own API token from the DISCOS website, and edit the file to include it. Alternatively, save the token in a plain .txt file in the project directory with filename ".esa_token.txt"

Usage example:

```python
> import sat_data                                                                                               
> df = sat_data.get_data(database = 'objects')                                                                  
> df.columns                                                                                                    
  Index(['DiscosID', 'IntlDes', 'XSectMin', 'XSectMax', 'SatName', 'Mass',
       'ObjectType', 'Height', 'Depth', 'XSectAvg', 'SatNo', 'Shape',
       'VimpelId', 'Length', 'LaunchId', 'OperatorId', 'ReentryId',
       'DestOrbitId', 'InitOrbitId'],
      dtype='object')
> df[df.SatNo == 1].SatName                                                                                     
  0    Sputnik (8K71PS) Blok-A
  Name: SatName, dtype: object
```
