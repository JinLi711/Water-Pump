|Input                   | Action                                                       | Justification |
| ---------------------- | ------------------------------------------------------------ | ------- |
|id                      | drop                                                      |all ids are unique
|amount_tsh              | binning, box cox, or log                                  |large outliers
|date_recorded           | convert this to number since first day                    |datetime itself is not useful
|funder                  | binary binning (well known vs not well know)               |                                    :1900 categories
|gps_height              | do something about the zeros                              |too many zeros and not sure if 0 values are possible
|installer               | binning                                                   |2145 categories
|longitude               | drop outliers                                             |has outliers of longitude=0
|latitude                | drop outliers                                             |has outliers of latitude=0
|wpt_name                | drop                                                      |37400 categories and intuitively has no relevancy
|num_private             | drop                                                      |96% corr with wpt_name, no description on what it is, and 3.8e-7 corr with labels                       
|basin                   | drop                                                      |99% corr with region and region has higher corr with labels
|subvillage              | drop                                                      |19000 categories
|region                  | NOTHING                                                      |
|region_code             | drop                                                      |100% corr with region
| district_code          | NOTHING                                                      |
|lga                     | binning, then check corr with region, latitude            |high corr with region and has many labels
|ward                    | binning, then check corr with lga, latitude, gps_height   |high corr with lga but not symmetric, also has many categories
|population              | drop outliers or binning                                  |very skewed
|public_meeting          | drop                                                      |7% corr with label
|recorded_by             | drop                                                      |there's only one value
|scheme_management       | NOTHING                                                  |
|scheme_name             | drop                                                     |2700 categories, 20000 (2/5 of data) are missing, also probably similar to scheme_management
|permit                  | drop                                                      |3% corr with label and is binary
|construction_year       | convert year to time, replace 0 by median                 |has 0 as entry values 
|extraction_type         | drop                                                      |100% corr with 'extraction_type_group' and 'extraction_type_class'
|extraction_type_group   | drop                                                      |100% corr with 'extraction_type_group' and class is least noisy
|extraction_type_class   | NOTHING                                                  |
|management              | NOTHING                                                   |  categories look the same as scheme_management, but the corr test was low between them and the categories have different values
|management_group        | drop                                                      |5% corr with label and 100% corr with management          
|payment                 | drop                                                      |100% corr with payment_type
|payment_type            | NOTHING                                                  |
|water_quality           | drop                                                      |100% corr with quality_group and has lower corr with label
|quality_group           | NOTHING                                                  |
|quantity                | drop                                                      | 100% corr with quantity group
|quantity_group          | NOTHING                                                  |
|source                  | NOTHING                                                  |
|source_type             | drop                                                        |100% corr with source
|source_class            | drop                                                      |7% corr with label, and trinary with low freq in one of the category, also 100% corr with source
|waterpoint_type         | drop                                                      |100% corr with water_type_group and more noisy
|waterpoint_type_group   | NOTHING                                                      |