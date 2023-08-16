# NolenLab
	Name: 
		tirf_patch_analysis_v4

	Author: 
		Adam Fries, Mike Lynch, Heidy Narvaez Ortiz

	Affiliation: 
		University of Oregon

	Description:
		Aligns track results from TrackMate from the end of one tracked channel 
		to the maximum intensity of a second tracked channel. For each tracked 
		channel calulates the length the track distance from the track origin. 
		Calulates the intensity accumulation and de-accumulation rates of tracks 
		given a user defined time window. Rates are clculated using the raw intensity 
		data. Plots of track data are smoothed using a 3-point average, normalized and 
		temporally offset relative to the maximum track intensity of the selected track.

	Input Parameters:
		The program uses a GUI interface for input parameters - 

		GUI Input Fields:
			1. Las17 csv path (aka Green) - Las17 Trackmate csv output
			2. ABP1 csv path (aka Red) - ABP1 Trachmate csv output
			3. Pair csv path - colocalization track data for Las17 and ABP1 to reference 
								and match the pairs of track IDs of interest 
			4. Save path - path to save output results
			5. Image Time Scale (s) - time scale from the microscope imaging
			6. Accumulation Time Window from Track Maximum Intensity - number of time points
			7. De-accumulation Time Window from Track Maximum Intensity - number of time points

	Output:
		A. csv files for each channel containing the following columns of data - 
			1. Track ID	
			2. Pair Number	
			3. Raw Intensity	
			4. Smooth Intensity	
			5. Distance from Origin	
			6. Smooth Distance from Origin	
			7. Delta Distance	
			8. Speed	
			9. Frame	
			10. Offset	
			11. Time Scale	
			12. Lifetime
		B. png files of the plot of the track intensities and distance from origin vs. time
			and include the accumulation rates in the plot legends
