# Rb-Sr-Iolite4
Rb-Sr Data Reduction Schemes and Isochron Tools for Iolite4

Here are the first versions of the iolite4 DRSs for Neoma, Quad, and the Isochron Tools. The Neoma and/or Quad DRSs go in your Plugins-> Data Reduction Schemes Folder. The Isochron Tool goes in your Plugins->User interface folder. You can check your paths in iolite preferences to make sure that you are putting things in the correct place for your setup.

It is recommended that you run the DRS on your data before using the Isochron Tools, though it should be able to find the channels you create in your own DRS. You will need to assign your secondary reference mica or feldspar for the age correction as type=reference material. .json files are included for some common materials. The DRS will automatically pull in the age and intercept for these materials, though sometimes you have to choose a different material and then go back to the one you want in order to get the info to update in the DRS widget. Click the button in the upper right to refresh the results after you crunch your data. You can then export this table to excel so that you have all of your DRS and secondary age correction information in a nice publication format table. 

Then open the Isochron Tool from the Tools tab in iolite. It will default to the age-corrected Rb/Sr ratios, but you can choose to plot not-age-corrected data with the dropdown menus. There is a multi-panel function, and export options at the bottom.
