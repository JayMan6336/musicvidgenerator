import os
import argparse
from pydub import AudioSegment
import numpy as np
import yt_dlp
from moviepy import AudioFileClip

def download_youtube_audio(video_urls, output_folder):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': f'{output_folder}/%(title)s.%(ext)s',
        'ignoreerrors': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(video_urls)

    # Return the list of downloaded files
    return [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.mp3')]

# Constants
SUPPORTED_FORMATS = ('.mp3', '.wav', '.flac', '.aac', '.m4a', '.wma', '.ogg')
TARGET_FREQUENCY = 432.0
DEFAULT_TARGET_SAMPLE_RATE = 48000
DEFAULT_TARGET_BITRATE = "192k"

# The frequency of 440hz tuning
tone_freq = np.array([
    16.35, 17.32, 18.35, 19.45, 20.6, 21.83, 23.12, 24.5, 25.96, 27.5, 29.14, 30.87,
    32.7, 34.65, 36.71, 38.89, 41.2, 43.65, 46.25, 49, 51.91, 55, 58.27, 61.74,
    65.41, 69.3, 73.42, 77.78, 82.41, 87.31, 92.5, 98, 103.83, 110, 116.54, 123.47,
    130.81, 138.59, 146.83, 155.56, 164.81, 174.61, 185, 196, 207.65, 220, 233.08, 246.94,
    261.63, 277.18, 293.66, 311.13, 329.63, 349.23, 369.99, 392, 415.3, 440, 466.16, 493.88,
    523.25, 554.37, 587.33, 622.25, 659.25, 698.46, 739.99, 783.99, 830.61, 880, 932.33, 987.77,
    1046.5, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91, 1479.98, 1567.98, 1661.22, 1760, 1864.66, 1975.53,
    2093, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83, 2959.96, 3135.96, 3322.44, 3520, 3729.31, 3951.07,
    4186.01, 4434.92, 4698.63, 4978.03, 5274.04, 5587.65, 5919.91, 6271.93, 6644.88, 7040, 7458.62, 7902.13
])

# Generate frequency of each tuning
tones = {str(frequency): tone_freq * frequency / 440.0 for frequency in np.arange(424.0, 448.1, 0.1)}

def speed_change(sound, speed=1.0, target_sample_rate=DEFAULT_TARGET_SAMPLE_RATE):
    """Change the speed of the audio and resample to the target sample rate."""
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
         "frame_rate": int(sound.frame_rate * speed)
    })
    return sound_with_altered_frame_rate.set_frame_rate(target_sample_rate)

def analyze_tone(song):
    """Analyze the dominant frequency of the audio."""
    sample_rate = song.frame_rate
    duration = song.duration_seconds
    
    # Handle audio shorter than 100 seconds
    if duration < 101:
        pad_ms = int((120.0 - duration) * 1000)
        silence = AudioSegment.silent(duration=pad_ms)
        audio = song + silence
    else:
        audio = song
    
    # Get first 100 seconds of the audio
    first_100_second = audio[:100*1000].set_channels(1)  # Mono
    
    # Get raw data (samples) and FFT parameters
    samples = np.array(first_100_second.get_array_of_samples())
    samples_len = len(samples)
    timestep = 1.0 / float(first_100_second.frame_rate)
    freqstep = sample_rate / samples_len
    
    # Compute FFT
    fft_result = np.fft.fft(samples)
    fft_freq = np.fft.fftfreq(samples_len, d=timestep)
    fft_normed = np.abs(fft_result) / len(fft_result)

    # Find the tuning frequency
    max_sum = 0
    max_freq = 0
    for frequency in np.arange(424.0, 448.1, 0.1):
        tone = tones[str(frequency)]
        sum_freq = np.sum(fft_normed[np.round(tone / freqstep).astype(int)])
        if sum_freq > max_sum:
            max_sum = sum_freq
            max_freq = frequency

    return max_freq

def convert_to_432hz(input_file, output_folder=None, target_format="mp3", target_bitrate=DEFAULT_TARGET_BITRATE, target_sample_rate=DEFAULT_TARGET_SAMPLE_RATE):
    try:
        if not input_file.lower().endswith(SUPPORTED_FORMATS):
            print(f"Error: Unsupported file format for '{input_file}'. Supported formats are: {', '.join(SUPPORTED_FORMATS)}")
            return

        folder, filename = os.path.split(input_file)
        output_folder = output_folder or folder
        new_filename = f"432hz_{os.path.splitext(filename)[0]}.{target_format}"
        output_file = os.path.join(output_folder, new_filename)
        
        if os.path.exists(output_file):
            print(f"Output file '{output_file}' already exists. Skipping conversion.")
            return
        
        print(f"Converting: {input_file}")
        
        song = AudioSegment.from_file(input_file)
        
        # Analyze tone
        tone = analyze_tone(song)
        print(f"Detected tone: {tone:.1f}Hz")
        
        # Calculate the 432Hz speed ratio
        speed_ratio = TARGET_FREQUENCY / tone
        print(f"Using Speed Ratio {speed_ratio:.4f} to convert...")
        
        # Convert audio
        new_song = speed_change(song, speed_ratio, target_sample_rate)
        
        # Save audio
        print(f"Saving to {output_file}")
        new_song.export(output_file, format=target_format, bitrate=target_bitrate)
        
        print("Conversion complete.")
    except Exception as e:
        print(f"An error occurred while processing '{input_file}': {str(e)}")

def batch_convert_directory(directory, output_folder=None, target_format="mp3", target_bitrate=DEFAULT_TARGET_BITRATE, target_sample_rate=DEFAULT_TARGET_SAMPLE_RATE):
    """Convert all supported audio files in a directory to 432Hz."""
    files_to_convert = [f for f in os.listdir(directory) if f.lower().endswith(SUPPORTED_FORMATS) and not f.startswith("432hz_")]
    total_files = len(files_to_convert)
    
    print(f"Found {total_files} files to convert in {directory}")
    
    for i, filename in enumerate(files_to_convert, 1):
        input_file = os.path.join(directory, filename)
        print(f"Converting file {i} of {total_files}: {filename}")
        convert_to_432hz(input_file, output_folder, target_format, target_bitrate, target_sample_rate)
        print(f"Progress: {i}/{total_files} files converted")

    print("Batch conversion complete.")
    
def ensure_output_directory(output_folder):
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

def main():
    parser = argparse.ArgumentParser(description="432Hz Converter Command - Convert audio files or YouTube videos to 432Hz")
    parser.add_argument("input", help="Input file, directory, or YouTube URL(s)")
    parser.add_argument("-o", "--output", help="Output directory (default: same as input)")
    parser.add_argument("-f", "--format", default="mp3", help="Output format (default: mp3)")
    parser.add_argument("-b", "--bitrate", default=DEFAULT_TARGET_BITRATE, help=f"Output bitrate (default: {DEFAULT_TARGET_BITRATE})")
    parser.add_argument("-s", "--sample-rate", type=int, default=DEFAULT_TARGET_SAMPLE_RATE, help=f"Output sample rate (default: {DEFAULT_TARGET_SAMPLE_RATE})")
    parser.add_argument("--youtube", action="store_true", help="Treat input as YouTube URL(s)")
    
    args = parser.parse_args()
    
    if args.youtube:
        # Handle YouTube URL(s)
        video_urls = args.input.split(',') if ',' in args.input else [args.input]
        temp_folder = 'temp_youtube_audio'
        os.makedirs(temp_folder, exist_ok=True)
        downloaded_files = download_youtube_audio(video_urls, temp_folder)
        
        for file in downloaded_files:
            convert_to_432hz(file, args.output, args.format, args.bitrate, args.sample_rate)
        
        # Clean up temporary files
        for file in downloaded_files:
            os.remove(file)
        os.rmdir(temp_folder)
    elif os.path.isfile(args.input):
        convert_to_432hz(args.input, args.output, args.format, args.bitrate, args.sample_rate)
    elif os.path.isdir(args.input):
        batch_convert_directory(args.input, args.output, args.format, args.bitrate, args.sample_rate)
    else:
        print(f"Error: '{args.input}' is not a valid file, directory, or YouTube URL.")

if __name__ == "__main__":
    main()
