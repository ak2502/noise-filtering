## Noise Filtering

We are using 3 algorithms to perform noise filtering on test_noise.wav audio file which contains some background noise.

The algorithms used are:-
1) Time Domain - Moving Average Filter (moving_average.py)
2) Frequency Domain Filter (freq_filter.py)
3) Spectral Subtraction (spectral_subtraction.py)

To run this in your device:
1) Make sure you have Python along with all the imported libraries installed.
2) Enter the wav file 
3) Replace the line in code:
   `f = we.open("test_noise.wav", 'rb')` with `f = we.open("your_file_name.wav", 'rb')`
4) Run the python file using terminal or any other IDE (spyder, VS Code, etc)
5) The output wav file will be saved in the same folder which has your python file. 
