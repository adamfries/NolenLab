# import libraries
from appJar import gui
import numpy as np
import pandas as pd
import scipy.optimize, scipy.stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import copy

def lin_model(x, m, b):
	return m*x + b

def lin_model_r(param, x, obs):
	return lin_model(x, param[0], param[1]) - obs

def fit_model(x, obs, param_guesses = (1, 1)):
	fit = scipy.optimize.least_squares(lin_model_r, 
									   param_guesses, 
									   args = (x, obs))
	fit_m = fit.x[0]
	fit_b = fit.x[1]
	
	return fit_m, fit_b

def tirf_analysis(ch1spot_path, ch2spot_path, pair_path, save_path, tscale, accum_win, deccum_win):

	## grab the file names and use to save the data, 
	##	create a directory to save all pngs
	ch1name = os.path.splitext(os.path.basename(ch1spot_path))[0]
	ch2name = os.path.splitext(os.path.basename(ch2spot_path))[0]
	
	png_path = os.path.join(save_path, ch1name + '_' + ch2name + '_pngs')

	try:
		png_dir = os.mkdir(png_path)
	except FileExistsError:
		print('Save path: ' + png_path + 
				'/ exists! Please delete and rerun.')
		exit()

	## read in the csv files as pandas Data Frames
	df1_spots = pd.read_csv(ch1spot_path, 
							usecols = [2, 4, 5, 8, 16], 
							header = 1)

	df2_spots = pd.read_csv(ch2spot_path, 
							usecols = [2, 4, 5, 8, 16], 
							header = 1)

	df_pairs = pd.read_csv(pair_path, 
						   usecols = [1, 2], 
						   header = 1)


	## create empty dataframes to populate with results for plotting
	df_results_ch1 = pd.DataFrame(columns = ['Track ID', 
											'Ch1_Raw_Intensity',
											'Ch1_Norm_Intensity', 
											'Ch1_Smooth_Intensity', 
											'Ch1_DisplaceOrigin', 
											'Ch1_Smooth_DisplaceOrigin', 
											'Ch1_Deldistance', 
											'Ch1_Speed', 'Frame'])

	df_results_ch2 = pd.DataFrame(columns = ['Track ID', 
											'Ch2_Raw_Intensity', 
											'Ch2_Norm_Intensity',
											'Ch2_Smooth_Intensity', 
											'Ch2_DisplaceOrigin', 
											'Ch2_Smooth_DisplaceOrigin', 
											'Ch2_Deldistance', 
											'Ch2_Speed', 'Frame', 'Offsets'])

	## create empty dataframes to poplate with results for saving
	ch1csv = pd.DataFrame(columns = ['Track ID', 
									'Pair Number', 
									'Raw Intensity',
									'Norm Intensity',
									'Smooth Intensity',
									'Distance from Origin',
									'Smooth Distance from Origin',
									'Delta Distance', 'Speed',
									'Frame', 'Time Scale',
									'Max Raw Intensity'])

	ch2csv = pd.DataFrame(columns = ['Track ID', 
									'Pair Number', 
									'Raw Intensity',
									'Norm Intensity',
									'Smooth Intensity',
									'Distance from Origin',
									'Smooth Distance from Origin',
									'Delta Distance', 'Speed',
									'Frame', 'Offset', 'Time Scale', 
									'Max Raw Intensity'])


	df_offsets = pd.DataFrame(columns = ['Track ID', 'Offsets'])

	## define the window size for the moving average
	##	and the empty list to populate frame shifts 
	N = 3
	fns = []

	max_c1_size = 0
	max_c2_size = 0

	max_c1_end = 0
	max_c2_end = 0

	max_c1_pos = 0
	max_c2_pos = 0

	## for each channel in df_pairs, 
	##	find the matching trajectory by track ID in df#_spots
	for idx, col in enumerate(df_pairs.columns):
		for idy, tid in enumerate(df_pairs.loc[:, col]):
			## skip if the field is NaN
			if str(tid) != 'nan':
				## some instances get read in as floats for some reason
				##	convert to string, skip the current track in ch1 and its
				##	corresponding track in ch2 if the id can't be read in 
				try:
					tid = str(int(tid))
					skip = None	
				except ValueError:
					print('Warning: "', 
							tid, 
							'" is an invalid expression: skipping')
					skip = idy
					continue
					
				## grab the ch1 spots dataframe that matches 
				##	the current pair track id, each channel of tracks
				##	gets read in sequentially, idx == 0 is ch1, idx == 1 is
				##	ch2
				if idx == 0:
					df_xyint = df1_spots.loc[df1_spots['Track ID'] == int(tid)]
				elif idx == 1:
					if skip == None:
						df_xyint = df2_spots.loc[df2_spots['Track ID'] == int(tid)]
					elif skip != None and idy != skip:
						df_xyint = df2_spots.loc[df2_spots['Track ID'] == int(tid)]
					else:
						continue		

				## calculate the N-pt average using convolution, 
				##	requires at least 3 points
				df_int = df_xyint.loc[:, 'total intensity'].to_numpy()*1.0
				df_raw = copy.deepcopy(df_int)
				max_raw = max(df_raw)
				df_x = df_xyint.loc[:, 'X'].to_numpy()*1.0
				df_y = df_xyint.loc[:, 'Y'].to_numpy()*1.0
				df_frame = df_xyint.loc[:, 'frame'].to_numpy()*1.0

				## added lifetime calculation 2021.04.08
				df_lifetime = (df_frame[-1] - df_frame[0])*tscale

				try:
					three_pt = np.convolve(df_int, np.ones((N,)) / N,
						mode = 'valid')	
				except ValueError:
					print('Warning! Problem with Track ID:', tid, ':skipping')	
					print('Found an empty dataframe')
					continue	
					

				## basline subtraction and normalize to max
				df_int -= min(df_int)
				df_int /= max(df_int)


				## baseline subtraction of min value 
				three_pt -= min(three_pt)
		
				## normalize by dividing by max value
				##  but check for a divide by 0
				if max(three_pt) != 0:
					three_pt /= max(three_pt)	
				else:
					print('Warning! Problem with Track ID:', tid, ':skipping')
					print('Found a 3-pt maximum == 0 divide by 0!')
					continue
			
				## get the frame number for where the max int is (1)
				##	append to the offset frame value to a list 
				if idx == 1:
					fn = df_frame[list(df_int).index(1)]
					fns.append(fn)	

				## compute distances from origin
				dist = np.sqrt((df_x - df_x[0])**2 + (df_y - df_y[0])**2)
				
				## compute a 3 point moving average of the distance from origin
				try:
					d3_pt = np.convolve(dist, np.ones((N,)) / N,
						mode = 'valid')	
				except ValueError:
					print('Warning! Problem with Track ID:', tid, ':skipping')	
					print('Found an empty dataframe')
					continue				


				## compute the delta-distance
				dd = np.diff(tuple(zip(df_x, df_y)), axis = 0)
				del_dist = np.hypot(dd[:, 0], dd[:, 1])
				
				## compute the speed
				speed = np.diff(del_dist, axis = 0)
				
				## save the channel stats and shift both channels by the max of 
				##	channel 1: updated from dist to normalized max
				curr_max_pos = np.argwhere(df_int == np.max(df_int))
				
				if idx == 0:	

					## get the position of the max position and store it is max
					if len(curr_max_pos) > 1:
						print('Warning!: found multiple max peaks, skipping track', tid)
						continue
					else:
						if curr_max_pos > max_c1_pos:
							max_c1_pos = curr_max_pos
						## collect all the dist from origin aligned to last point
						if len(df_int) > max_c1_size:
							max_c1_size = len(df_int)
						## get the max distance from the max pos and end
						if (len(df_int) - curr_max_pos) > max_c1_end:
							max_c1_end = len(df_int) - curr_max_pos					


						df_results_ch1.loc[idy] = [tid, df_raw, df_int, three_pt, dist, 
													d3_pt, del_dist, speed, df_frame]
						
						d = {'Track ID': np.array(tid), 
								'Pair Number': np.array(idy),
								'Raw Intensity': np.array(df_raw),
								'Norm Intensity': np.array(df_int), 
								'Smooth Intensity': np.array(three_pt), 
								'Distance from Origin': np.array(dist), 
								'Smooth Distance from Origin': np.array(d3_pt), 
								'Delta Distance': np.array(del_dist),
								'Speed': np.array(speed), 
								'Frame': np.array(df_frame), 
								'Time Scale': np.array(tscale), 
								'Lifetime (s)': np.array(df_lifetime),
								'Max Raw Intensity': np.array(max_raw)
								}
						
						ch1outcsv = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
						ch1csv = pd.concat([ch1csv, ch1outcsv])

				else:
					## get the position of the max position and store it is max
					if len(curr_max_pos) > 1:
						print('Warning!: found multiple max peaks, skipping track', tid)
						continue
					else:
						if curr_max_pos > max_c2_pos:
							max_c2_pos = curr_max_pos
						## collect all the normaized int aligned to last point
						if len(df_int) > max_c2_size:
							max_c2_size = len(df_int)
						## get the max distance from the max pos and end
						if (len(df_int) - curr_max_pos) > max_c2_end:
							max_c2_end = len(df_int) - curr_max_pos					
										

						df_results_ch2.loc[idy] = [tid, df_raw, df_int, three_pt, dist, 
													d3_pt, del_dist, speed, df_frame, fns]
						
						d = {'Track ID': np.array(tid),
								'Pair Number': np.array(idy), 
								'Raw Intensity': np.array(df_raw),
								'Norm Intensity': np.array(df_int), 
								'Smooth Intensity': np.array(three_pt), 
								'Distance from Origin': np.array(dist), 
								'Smooth Distance from Origin': np.array(d3_pt), 
								'Delta Distance': np.array(del_dist),
								'Speed': np.array(speed), 
								'Frame': np.array(df_frame), 
								'Offset': np.array(fn),
								'Time Scale': np.array(tscale),
								'Lifetime (s)': np.array(df_lifetime),
								'Max Raw Intensity': np.array(max_raw)
								}

					
						ch2outcsv = pd.DataFrame({k: pd.Series(v) for k, v in d.items()})
						ch2csv = pd.concat([ch2csv, ch2outcsv])				


	## save the results for ch1 and ch1 to csv files
	print('SAVING RESULTS TO: ', save_path)

	ch1csv.to_csv(os.path.join(save_path, ch1name + '_results.csv'))
	ch2csv.to_csv(os.path.join(save_path, ch2name + '_results.csv'))

	df_results_ch1ch2 = pd.concat([df_results_ch1, df_results_ch2], 
										axis = 1, ignore_index = True)

	ch1_int = df_results_ch1.loc[:, 'Ch1_Raw_Intensity'].to_numpy()
	ch2_int = df_results_ch2.loc[:, 'Ch2_Raw_Intensity'].to_numpy()

	ch1_distor = df_results_ch1.loc[:, 'Ch1_DisplaceOrigin'].to_numpy()
	ch2_distor = df_results_ch2.loc[:, 'Ch2_DisplaceOrigin'].to_numpy()

	## begin parsing end alignment distance from origin
	ch1_int_end_align = []
	ch2_int_end_align = []
	ch1_distor_end_align = []
	ch2_distor_end_align = []

	for i, arr in enumerate(ch1_int):
		diff = max_c1_size - len(arr)
		if diff > 0:
			ch1_int_end_align.append([np.nan]*int(diff) + list(arr))
			ch1_distor_end_align.append([np.nan]*int(diff) + list(ch1_distor[i]))
		else:
			ch1_int_end_align.append(arr)
			ch1_distor_end_align.append(ch1_distor[i])

	for i, arr in enumerate(ch2_int):
		diff = max_c2_size - len(arr)
		if diff > 0:
			ch2_int_end_align.append([np.nan]*int(diff) + list(arr))
			ch2_distor_end_align.append([np.nan]*int(diff) + list(ch2_distor[i]))
		else:
			ch2_int_end_align.append(arr)
			ch2_distor_end_align.append(ch2_distor[i])


	## 2021.05.28 use raw data instead of normalized to find accumulation
	##	rate and deaccumulation rate - also include the ability to parse
	##	the desired number of points from max
	

	## make these dataframes with the distance from origin included as dicts
	c1_align_end_df = pd.DataFrame({'Ch1 Intensity': ch1_int_end_align, 'Ch1 Dist from Origin (um)': ch1_distor_end_align})
	c2_align_end_df = pd.DataFrame({'Ch2 Intensity': ch2_int_end_align, 'Ch2 Dist from Origin (um)': ch2_distor_end_align})
	
	c1_align_end_df.to_csv(os.path.join(save_path, ch1name + '_end_align.csv'))
	c2_align_end_df.to_csv(os.path.join(save_path, ch2name + '_end_align.csv'))
	
	''' begin max alignment: all arrays should be relative to the longest array.
		we will prepend all arrays based on the greatest max position and append
		all arrays based on the greatest distance between the max position and the 
		length on the array '''
	## for channel 1
	for i, arr in enumerate(ch1_int):
		curr_max_pos = np.argwhere(arr == max(arr))
		prepend_num = int(np.abs(max_c1_pos - curr_max_pos))
		curr_c1_end = len(arr) - curr_max_pos
		append_num = int(np.abs(max_c1_end - curr_c1_end))				
		## pad the arrs
		pad_arr = np.pad(arr, (prepend_num, append_num), mode = 'constant')
		pad_dist = np.pad(ch1_distor[i], (prepend_num, append_num), mode = 'constant')

		## roll the arrs to align the max
		## first get the position of the new max
		if i == 0:
			static_max_pos = int(np.argwhere(pad_arr == np.max(pad_arr)))
			pad_arr[pad_arr == 0] = np.nan
			pad_dist[pad_dist == 0] = np.nan
			max_align_ch1 = [list(pad_arr)]
			max_align_dist1 = [list(pad_dist)]
		else:
			curr_max_pos = int(np.argwhere(pad_arr == np.max(pad_arr)))
			pad_arr = np.roll(pad_arr, static_max_pos - curr_max_pos)
			pad_dist = np.roll(pad_dist, static_max_pos - curr_max_pos)	
			pad_arr[pad_arr == 0] = np.nan
			pad_dist[pad_dist == 0] = np.nan
			max_align_ch1.append(list(pad_arr))
			max_align_dist1.append(list(pad_dist))
	
	## now for channel 2			
	for i, arr in enumerate(ch2_int):
		curr_max_pos = np.argwhere(arr == max(arr))
		prepend_num = int(np.abs(max_c2_pos - curr_max_pos))
		curr_c2_end = len(arr) - curr_max_pos
		append_num = int(np.abs(max_c2_end - curr_c2_end))				
		## pad the arrs
		pad_arr = np.pad(arr, (prepend_num, append_num), mode = 'constant')
		pad_dist = np.pad(ch2_distor[i], (prepend_num, append_num), mode = 'constant')

		## roll the arrs to align the max
		## first get the position of the new max
		if i == 0:
			static_max_pos = int(np.argwhere(pad_arr == np.max(pad_arr)))
			pad_arr[pad_arr == 0] = np.nan
			pad_dist[pad_dist == 0] = np.nan
			max_align_ch2 = [list(pad_arr)]
			max_align_dist2 = [list(pad_dist)]
		else:
			curr_max_pos = int(np.argwhere(pad_arr == np.max(pad_arr)))
			pad_arr = np.roll(pad_arr, static_max_pos - curr_max_pos)	
			pad_dist = np.roll(pad_dist, static_max_pos - curr_max_pos)	
			pad_arr[pad_arr == 0] = np.nan
			pad_dist[pad_dist == 0] = np.nan
			max_align_ch2.append(list(pad_arr))
			max_align_dist2.append(list(pad_dist))
	
	## plot the results and save the figures
	rcParams['font.family'] = 'Times New Roman'	
	rcParams['font.size'] = 16

	del_time = []
	## loop through and plot the dual channel scatter plots
	for i in range(df_results_ch1ch2.shape[0]):
		ch1_frame = (df_results_ch1.loc[:, 'Frame'].to_numpy())[i]
		ch2_frame = (df_results_ch2.loc[:, 'Frame'].to_numpy())[i]
		ch1_raw = (df_results_ch1.loc[:, 'Ch1_Raw_Intensity'].to_numpy())[i]
		ch1_norm = (df_results_ch1.loc[:, 'Ch1_Norm_Intensity'].to_numpy())[i]
		ch2_norm = (df_results_ch2.loc[:, 'Ch2_Norm_Intensity'].to_numpy())[i]
		ch1_smooth = (df_results_ch1.loc[:, 'Ch1_Smooth_Intensity'].to_numpy())[i]
		ch2_raw = (df_results_ch2.loc[:, 'Ch2_Raw_Intensity'].to_numpy())[i]
		ch2_smooth = (df_results_ch2.loc[:, 'Ch2_Smooth_Intensity'].to_numpy())[i]
		ch1_labels = (df_results_ch1.loc[:, 'Track ID'].to_numpy())[i]
		ch2_labels = (df_results_ch2.loc[:, 'Track ID'].to_numpy())[i]
		ch1_dist_or = (df_results_ch1.loc[:, 'Ch1_DisplaceOrigin'].to_numpy())[i]
		ch2_dist_or = (df_results_ch2.loc[:, 'Ch2_DisplaceOrigin'].to_numpy())[i]
		ch1_sm_dist_or = (df_results_ch1.loc[:, 'Ch1_Smooth_DisplaceOrigin'].to_numpy())[i]
		ch2_sm_dist_or = (df_results_ch2.loc[:, 'Ch2_Smooth_DisplaceOrigin'].to_numpy())[i]
		

		time1 = (ch1_frame - fns[i])*tscale
		time2 = (ch2_frame - fns[i])*tscale
		time_sm1 = (ch1_frame[1:-1] - fns[i])*tscale
		time_sm2 = (ch2_frame[1:-1] - fns[i])*tscale


		del_time.append([abs(time1[-1] - time2[0]), ch1_labels, ch2_labels])

		fig, ax = plt.subplots(2, 1, figsize = (10,8), sharex = True)
		ax[1].plot(time1, ch1_norm, 'og', 
			label = 'Track ' + ch1_labels, alpha = 0.35)
		ax[1].plot(time_sm1, ch1_smooth, 'g', 
			label = str(N) + ' pt avg.')
		ax[1].plot(time2, ch2_norm, 'or', 
			label = 'Track ' + ch2_labels, alpha = 0.35)
		ax[1].plot(time_sm2, ch2_smooth, 'r', 
			label = str(N) + ' pt avg.') 

		ax[0].plot(time1, ch1_dist_or, 'og', alpha = 0.35)
		ax[0].plot(time2, ch2_dist_or, 'or', alpha = 0.35)
		ax[0].plot(time_sm1, ch1_sm_dist_or, 'g')
		ax[0].plot(time_sm2, ch2_sm_dist_or, 'r')

		ax[1].set_xlabel('Shifted Time (s)')
		ax[1].set_ylabel('Normalized Intensity')
		ax[0].set_ylabel('Distance from Origin (um)')
		plt.subplots_adjust(hspace=.0)
		plt.legend(bbox_to_anchor = (1.1, 0.6), bbox_transform=plt.gcf().transFigure)	
		ax[0].grid()
		ax[1].grid()
		fig.savefig(os.path.join(png_path, ch1_labels + '_' + ch2_labels + '.png'), 
			bbox_inches='tight')
		plt.close(fig)

	## plot the combined shifted plots
	ch1_frame = df_results_ch1.loc[:, 'Frame'].to_numpy()
	ch2_frame = df_results_ch2.loc[:, 'Frame'].to_numpy()
	ch1_norm = df_results_ch1.loc[:, 'Ch1_Norm_Intensity'].to_numpy()
	ch2_norm = df_results_ch2.loc[:, 'Ch2_Norm_Intensity'].to_numpy()
	ch1_dist = df_results_ch1.loc[:, 'Ch1_DisplaceOrigin'].to_numpy()
	ch2_dist = df_results_ch2.loc[:, 'Ch2_DisplaceOrigin'].to_numpy()
	ch1_rawrfu = df_results_ch1.loc[:, 'Ch1_Raw_Intensity'].to_numpy()
	ch2_rawrfu = df_results_ch2.loc[:, 'Ch2_Raw_Intensity'].to_numpy()	

	time1_all = []
	time2_all = []
	ch1_all = []
	ch2_all = []	
	dist1_all = []
	dist2_all = []
	ch1_raw_all = []
	ch2_raw_all = []

	figall, axall = plt.subplots(2, 1, figsize = (10, 8), sharex = True)
	for i in range(len(ch1_frame)):
		
		time1 = list((ch1_frame[i] - fns[i])*tscale)
		time2 = list((ch2_frame[i] - fns[i])*tscale)
		ch1 = list(ch1_norm[i])
		ch2 = list(ch2_norm[i])
		dist1 = list(ch1_dist[i])
		dist2 = list(ch2_dist[i])
		ch1_allraw = list(ch1_rawrfu[i])
		ch2_allraw = list(ch2_rawrfu[i])

		## collect all data to save to csv
		time1_all.extend(time1)
		time2_all.extend(time2)
		ch1_all.extend(ch1)
		ch2_all.extend(ch2)		
		dist1_all.extend(dist1)
		dist2_all.extend(dist2)
		#ch1_raw_all.extend{ch1_allraw}
		#ch2_raw_all.extend{ch2_allraw}
		ch1_raw_all.extend(ch1_allraw)
		ch2_raw_all.extend(ch2_allraw)



		if i == len(ch1_frame) - 1:	
			axall[0].plot(time1, dist1, 'o', color = 'green', alpha = 0.4, label = 'Las17')
			axall[0].plot(time2, dist2, 'o', color = 'red', alpha = 0.4, label = 'ABP1')
		
			axall[1].plot(time1, ch1, 'o', color = 'green', alpha = 0.4, label = 'Las17')
			axall[1].plot(time2, ch2, 'o', color = 'red', alpha = 0.4, label = 'ABP1')
		else:
			axall[0].plot(time1, dist1, 'o', color = 'green', alpha = 0.4)
			axall[0].plot(time2, dist2, 'o', color = 'red', alpha = 0.4)
		
			axall[1].plot(time1, ch1, 'o', color = 'green', alpha = 0.4)
			axall[1].plot(time2, ch2, 'o', color = 'red', alpha = 0.4)

	axall[1].set_xlabel('Time (s)')
	axall[1].set_ylabel('Normalized Intensity')
	axall[0].set_ylabel('Distance From Origin (um)')
	plt.subplots_adjust(hspace=.0)
	plt.legend(bbox_to_anchor = (1.1, 0.6), bbox_transform=plt.gcf().transFigure)	
	axall[0].grid()
	axall[1].grid()

	figall.savefig(os.path.join(png_path, 'max_aligned_all.png'), bbox_inches='tight')

	## create data frame and save to csv, need to include distances
	all_ch1_time1 = pd.DataFrame({'All Time Ch1 (s)': np.array(time1_all), 'All Intensities Ch1': np.array(ch1_all), 'Distance From Origin Ch1 (um)': dist1_all, 'All Raw Intensities Ch1': np.array(ch1_raw_all)})
	all_ch2_time2 = pd.DataFrame({'All Time Ch2 (s)': np.array(time2_all), 'All Intensities Ch2': np.array(ch2_all), 'Distance From Origin Ch2 (um)': dist2_all, 'All Raw Intensities Ch2': np.array(ch2_raw_all)})
	
	all_ch1_time1.to_csv(os.path.join(save_path, ch1name + '_all.csv'))
	all_ch2_time2.to_csv(os.path.join(save_path, ch2name + '_all.csv'))
	
	
	## plot int max aligned plots
	figalign, axalign = plt.subplots(2, 1, figsize = (10, 10), sharex = True)
	
	max_align_ch1 = np.array(max_align_ch1) 
	max_align_ch2 = np.array(max_align_ch2)
	max_align_ch1 = np.where(~np.isnan(max_align_ch1), max_align_ch1, 0)
	max_align_ch2 = np.where(~np.isnan(max_align_ch2), max_align_ch2, 0)
	mean_max_align_ch1 = np.mean(max_align_ch1, axis = 0)
	mean_max_align_ch2 = np.mean(max_align_ch2, axis = 0)
	std_ch1 = np.std(max_align_ch1, axis = 0)
	std_ch2 = np.std(max_align_ch2, axis = 0)
	x1 = np.arange(len(list(mean_max_align_ch1)))*tscale
	x2 = np.arange(len(list(mean_max_align_ch2)))*tscale

	## slice out the accumulation to max
	max_pos1 = int(np.argwhere(mean_max_align_ch1 == np.max(mean_max_align_ch1)))
	max_pos2 = int(np.argwhere(mean_max_align_ch2 == np.max(mean_max_align_ch2)))

	accum_mean_align1 = mean_max_align_ch1[:max_pos1 + 1]
	accum_mean_align2 = mean_max_align_ch2[:max_pos2 + 1]
	
	accum_x1 = x1[:max_pos1 + 1]
	accum_x2 = x2[:max_pos2 + 1]
	
	am1, ab1 = fit_model(accum_x1[-accum_win:], accum_mean_align1[-accum_win:])
	am2, ab2 = fit_model(accum_x2[-accum_win:], accum_mean_align2[-accum_win:])

	daccum_mean_align1 = mean_max_align_ch1[max_pos1:]
	daccum_mean_align2 = mean_max_align_ch2[max_pos2:]
	
	daccum_x1 = x1[max_pos1:]
	daccum_x2 = x2[max_pos2:]

	dm1, db1 = fit_model(daccum_x1[:deccum_win], daccum_mean_align1[:deccum_win])
	dm2, db2 = fit_model(daccum_x2[:deccum_win], daccum_mean_align2[:deccum_win])
	paramsa1 = [am1, ab1]
	paramsa2 = [am2, ab2]
	paramsd1 = [dm1, db1]
	paramsd2 = [dm2, db2]

	divider1 = make_axes_locatable(axalign[0])
	divider2 = make_axes_locatable(axalign[1])
	res_ax1 = divider1.append_axes('bottom', size = '30%', pad = 0, sharex = axalign[0])
	res_ax2 = divider2.append_axes('bottom', size = '30%', pad = 0, sharex = axalign[0])


	
	axalign[0].plot(x1, mean_max_align_ch1, '-g', label = 'avg. Las17')
	axalign[0].plot(accum_x1[-accum_win:], lin_model(accum_x1[-accum_win:], am1, ab1), '--k', label = 'm_a = {:.3f}'.format(am1) + ' s^{-1}')
	axalign[0].plot(daccum_x1[:deccum_win], lin_model(daccum_x1[:deccum_win], dm1, db1), ':k', label = 'm_d = {:.3f}'.format(dm1) + ' s^{-1}')
	res_ax1.plot(accum_x1[-accum_win:], lin_model_r(paramsa1, accum_x1[-accum_win:], accum_mean_align1[-accum_win:]), '--k')	
	res_ax1.plot(daccum_x1[:deccum_win], lin_model_r(paramsd1, daccum_x1[:deccum_win], daccum_mean_align1[:deccum_win]), ':k')
	plt.setp(axalign[1].get_xticklabels(), visible = False)
	plt.setp(res_ax1.get_xticklabels(), visible = False)
	res_ax1.grid()

	
	
	axalign[1].plot(x2, mean_max_align_ch2, '-r', label = 'avg. ABP1')
	axalign[1].plot(accum_x2[-accum_win:], lin_model(accum_x2[-accum_win:], am2, ab2), '--k', label = 'm_a = ' + '{:.3f}'.format(am2) + ' s^{-1}')
	axalign[1].plot(daccum_x2[:deccum_win], lin_model(daccum_x2[:deccum_win], dm2, db2), ':k', label = 'm_d = ' + '{:.3f}'.format(dm2) + ' s^{-1}')	
	res_ax2.plot(accum_x2[-accum_win:], lin_model_r(paramsa2, accum_x2[-accum_win:], accum_mean_align2[-accum_win:]), '--k')	
	res_ax2.plot(daccum_x2[:deccum_win], lin_model_r(paramsd2, daccum_x2[:deccum_win], daccum_mean_align2[:deccum_win]), ':k')
	res_ax2.set_xlim([np.min(accum_x1), np.max(daccum_x1)])
	res_ax2.grid()
	

	axalign[0].fill_between(x1, mean_max_align_ch1 - std_ch1, mean_max_align_ch1 + std_ch1, color = 'g', alpha = 0.35, label = 'std. Las17')
	axalign[1].fill_between(x2, mean_max_align_ch2 - std_ch2, mean_max_align_ch2 + std_ch2, color = 'r', alpha = 0.35, label = 'std. ABP1')
	
	
	axalign[0].set_ylabel('Raw Intensity Las17')
	axalign[1].set_ylabel('Raw Intensity ABP1')
	res_ax2.set_xlabel('Max-Aligned Time (s)')
	res_ax1.set_ylabel('Resid.')
	res_ax2.set_ylabel('Resid.')
	axalign[0].grid()
	axalign[1].grid()
	handles0, labels0 = axalign[0].get_legend_handles_labels()
	handles1, labels1 = axalign[1].get_legend_handles_labels()
	
	plt.subplots_adjust(hspace = .0)
	axalign[0].legend(handles0, labels0, bbox_to_anchor = (1, 0.680), bbox_transform=plt.gcf().transFigure)
	axalign[1].legend(handles1, labels1, bbox_to_anchor = (1.25, 0.500), bbox_transform=plt.gcf().transFigure)
	
	figalign.savefig(os.path.join(png_path, 'max_aligned_intensity.png'), bbox_inches='tight')
	plt.close(fig)

	## plot the max aligned distance from origin
	figalignd, axalignd = plt.subplots(2, 1, figsize = (10, 10), sharex = True)

	
	max_align_d1 = np.array(max_align_dist1) 
	max_align_d2 = np.array(max_align_dist2)
	max_align_d1 = np.where(~np.isnan(max_align_d1), max_align_d1, 0)
	max_align_d2 = np.where(~np.isnan(max_align_d2), max_align_d2, 0)
	mean_max_align_d1 = np.mean(max_align_d1, axis = 0)
	mean_max_align_d2 = np.mean(max_align_d2, axis = 0)
	std_d1 = np.std(max_align_d1, axis = 0)
	std_d2 = np.std(max_align_d2, axis = 0)

	
	axalignd[0].plot(x1, mean_max_align_d1, '-g', label = 'avg. Las17')
	axalignd[1].plot(x2, mean_max_align_d2, '-r', label = 'avg. ABP1')
	axalignd[0].fill_between(x1, mean_max_align_d1 - std_d1, mean_max_align_d1 + std_d1, color = 'g', alpha = 0.35, label = 'std. Las17')
	axalignd[1].fill_between(x2, mean_max_align_d2 - std_d2, mean_max_align_d2 + std_d2, color = 'r', alpha = 0.35, label = 'std. ABP1')
	
	
	axalignd[0].set_ylabel('Distance from Origin Las17')
	axalignd[1].set_ylabel('Distance from Origin ABP1')
	axalignd[1].set_xlabel('Max-Aligned Time (s)')
	axalignd[0].grid()
	axalignd[1].grid()
	plt.subplots_adjust(hspace = .0)
	axalignd[0].legend(bbox_to_anchor = (1, 0.680), bbox_transform=plt.gcf().transFigure)
	axalignd[1].legend(bbox_to_anchor = (1.25, 0.500), bbox_transform=plt.gcf().transFigure)
	
	figalignd.savefig(os.path.join(png_path, 'max_aligned_distorigin.png'), bbox_inches='tight')
	plt.close(fig)




	## transpose the data to be easily read in Prism and save to fule
	c1_align_max_df = pd.DataFrame(max_align_ch1).T
	c2_align_max_df = pd.DataFrame(max_align_ch2).T
	dist1_align_max_df = pd.DataFrame(max_align_dist1).T
	dist2_align_max_df = pd.DataFrame(max_align_dist2).T

	c1_align_max_df.columns = list(df_results_ch1.loc[:, 'Track ID'])
	c2_align_max_df.columns = list(df_results_ch2.loc[:, 'Track ID'])
	dist1_align_max_df.columns = list(df_results_ch1.loc[:, 'Track ID'])
	dist2_align_max_df.columns = list(df_results_ch2.loc[:, 'Track ID'])
	

	## add the accumulation and decumulation slopes to the df
	c1_align_max_df['m_accumulation1'] = pd.Series(am1, index = c1_align_max_df.index[[0]])
	c1_align_max_df['m_decumulation1'] = pd.Series(dm1, index = c1_align_max_df.index[[0]])
	c2_align_max_df['m_accumulation2'] = pd.Series(am2, index = c2_align_max_df.index[[0]])
	c2_align_max_df['m_decumulation2'] = pd.Series(dm2, index = c2_align_max_df.index[[0]])
	
	c1_align_max_df.to_csv(os.path.join(save_path, ch1name + '_max_align_intensity.csv'))
	c2_align_max_df.to_csv(os.path.join(save_path, ch2name + '_max_align_intensity.csv'))
	dist1_align_max_df.to_csv(os.path.join(save_path, ch1name + '_max_align_distorigin.csv'))
	dist2_align_max_df.to_csv(os.path.join(save_path, ch2name + '_max_align_distorigin.csv'))
	


	## plot int end aligned plot
	figend1, axend1 = plt.subplots(2, 1, figsize = (10, 8), sharex = True)

	end_align_ch1 = np.array(ch1_int_end_align) 
	end_align_ch2 = np.array(ch2_int_end_align)
	end_align_dist1 = np.array(ch1_distor_end_align)
	end_align_dist2 = np.array(ch2_distor_end_align)

	mean_end_align_ch1 = np.nanmean(end_align_ch1, axis = 0)
	mean_end_align_ch2 = np.nanmean(end_align_ch2, axis = 0)
	mean_distor_ch1 = np.nanmean(end_align_dist1, axis = 0)
	mean_distor_ch2 = np.nanmean(end_align_dist2, axis = 0)

	std_ch1 = np.nanstd(end_align_ch1, axis = 0)
	std_ch2 = np.nanstd(end_align_ch2, axis = 0)
	std_dist1 = np.nanstd(end_align_dist1, axis = 0)
	std_dist2 = np.nanstd(end_align_dist2, axis = 0)

	x1 = np.arange(len(list(mean_end_align_ch1)))*tscale
	x2 = np.arange(len(list(mean_end_align_ch2)))*tscale
	
	axend1[0].plot(x1, mean_distor_ch1, '-g', label = 'avg. Las17')
	axend1[1].plot(x1, mean_end_align_ch1, '-g')
	axend1[0].fill_between(x1, mean_distor_ch1 - std_dist1, mean_distor_ch1 + std_dist1, color = 'g', alpha = 0.35, label = 'std. Las17')
	axend1[1].fill_between(x1, mean_end_align_ch1 - std_ch1, mean_end_align_ch1 + std_ch1, color = 'g', alpha = 0.35)
	axend1[0].set_ylabel('Distance From Origin (um)')
	axend1[1].set_ylabel('Raw Intensity')
	axend1[1].set_xlabel('End-Aligned Time (s)')
	axend1[0].grid()
	axend1[1].grid()
	handles0, labels0 = axend1[0].get_legend_handles_labels()
	handles1, labels1 = axend1[1].get_legend_handles_labels()
	

	plt.subplots_adjust(hspace = .0)
	axend1[0].legend(handles0, labels0, bbox_to_anchor = (1.1, 0.605), bbox_transform=plt.gcf().transFigure)
	axend1[1].legend(handles1, labels1, bbox_to_anchor = (1.1, 0.505), bbox_transform=plt.gcf().transFigure)
	
	figend1.savefig(os.path.join(png_path, 'end_aligned_Las17.png'), bbox_inches='tight')
	plt.close(fig)

	## begin ch2 end aligned dist and int plot
	figend2, axend2 = plt.subplots(2, 1, figsize = (10, 8), sharex = True)

	axend2[0].plot(x2, mean_distor_ch2, '-r', label = 'avg. ABP1')
	axend2[1].plot(x2, mean_end_align_ch2, '-r')
	axend2[0].fill_between(x2, mean_distor_ch2 - std_dist2, mean_distor_ch2 + std_dist2, color = 'r', alpha = 0.35, label = 'std. ABP1')
	axend2[1].fill_between(x2, mean_end_align_ch2 - std_ch2, mean_end_align_ch2 + std_ch2, color = 'r', alpha = 0.35)
	axend2[0].set_ylabel('Distance From Origin (um)')
	axend2[1].set_ylabel('Raw Intensity')
	axend2[1].set_xlabel('End-Aligned Time (s)')
	axend2[0].grid()
	axend2[1].grid()
	handles0, labels0 = axend2[0].get_legend_handles_labels()
	handles1, labels1 = axend2[1].get_legend_handles_labels()
	

	plt.subplots_adjust(hspace = .0)
	axend2[0].legend(handles0, labels0, bbox_to_anchor = (1.1, 0.605), bbox_transform=plt.gcf().transFigure)
	axend2[1].legend(handles1, labels1, bbox_to_anchor = (1.1, 0.505), bbox_transform=plt.gcf().transFigure)
	
	figend2.savefig(os.path.join(png_path, 'end_aligned_ABP1.png'), bbox_inches='tight')
	plt.close(fig)


	## transpose the data to be easily read in Prism and save to fule
	c1_align_end_df = pd.DataFrame(end_align_ch1).T
	dist1_align_end_df = pd.DataFrame(end_align_dist1).T
	dist2_align_end_df = pd.DataFrame(end_align_dist2).T
	c2_align_end_df = pd.DataFrame(end_align_ch2).T

	c1_align_end_df.columns = list(df_results_ch1.loc[:, 'Track ID'])
	c2_align_end_df.columns = list(df_results_ch2.loc[:, 'Track ID'])
	dist1_align_end_df.columns = list(df_results_ch1.loc[:, 'Track ID'])
	dist2_align_end_df.columns = list(df_results_ch2.loc[:, 'Track ID'])
	
	c1_align_end_df.to_csv(os.path.join(save_path, ch1name + '_end_align_intensity.csv'))
	c2_align_end_df.to_csv(os.path.join(save_path, ch2name + '_end_align_intensity.csv'))
	dist1_align_end_df.to_csv(os.path.join(save_path, ch1name + '_end_align_distorigin.csv'))
	dist2_align_end_df.to_csv(os.path.join(save_path, ch2name + '_end_align_distorigin.csv'))
	

	del_time = np.array(del_time)
	del_time_df = pd.DataFrame.from_dict({'Delta Time (s)': del_time[:, 0], 
										  'Track Green': del_time[:, 1], 
										  'Track Red': del_time[:, 2]})
	del_time_df.to_csv(os.path.join(save_path, 'red_start_green_end.csv'))	

def press(button):
	if button == 'Cancel':
		app.stop()
	else:
		fpath1 = app.getEntry('f1')
		fpath2 = app.getEntry('f2')
		fpath3 = app.getEntry('f3')
		fpath4 = app.getEntry('f4')
		tscale = app.getEntry('Time Scale (s)')
		accum_win = app.getEntry('Accumulation Window from Max')
		deccum_win = app.getEntry('De-accumulation Window from Max')

		#fpath2 = 'AllTraj_ABP1-Table skeleton.csv'
		#fpath1 = 'AllTraj_Las17-Table skeleton.csv'
		#fpath3 = 'Pairing-Table_Arp2_A207C.csv'
		#fpath4 = 'results_new/'

		## check that all path fields are not empty
		if len(fpath1) == 0 or len(fpath2) == 0 or len(fpath3) == 0 or len(fpath4) == 0:
			print('Error! Empty fields. Exiting')	
			exit()
		else:
			## if everything checks out, run the code
			tirf_analysis(fpath1, 
						  fpath2, 
						  fpath3, 
						  fpath4, 
						  float(tscale), 
						  int(accum_win), 
						  int(deccum_win))

# create a GUI variable called app
app = gui('Login Window', '600x400')
app.setBg('blue')
app.setFont(18)

app.addLabel('title', 'Welcome to TIRF Patch Analysis')
app.setLabelBg('title', 'black')
app.setLabelFg('title', 'white')


app.addFileEntry('f1')
app.addFileEntry('f2')
app.addFileEntry('f3')
app.addDirectoryEntry('f4')
app.addLabelEntry('Time Scale (s)')
app.addLabelEntry('Accumulation Window from Max')
app.addLabelEntry('De-accumulation Window from Max')
app.setLabelFg('Time Scale (s)', 'white')
app.setLabelFg('De-accumulation Window from Max', 'white')
app.setLabelFg('Accumulation Window from Max', 'white')

app.setEntryDefault('f1', '-- choose Las17 csv path --')
app.setEntryDefault('f2', '-- choose ABP1 csv path --')
app.setEntryDefault('f3', '-- choose Pair csv path --')
app.setEntryDefault('f4', '-- choose Save path --')
app.setEntryDefault('Time Scale (s)', '1.0')
app.setEntry('Time Scale (s)', '1.0')
app.setEntryDefault('Accumulation Window from Max', '20')
app.setEntryDefault('De-accumulation Window from Max', '20')
app.setEntry('Accumulation Window from Max', '20')
app.setEntry('De-accumulation Window from Max', '20')

app.addButtons(['Submit', 'Cancel'], press)

app.go()
