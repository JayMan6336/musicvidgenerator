import os
import argparse
from pydub import AudioSegment  # Used for audio processing
import numpy as np
import yt_dlp  # Used for platform downloads

def download_platform_audio(video_urls, output_folder):
    """Download audio from supported platforms (YouTube, BitChute, Rumble)"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '320',
        }],
        'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
        'ignoreerrors': True,
        'no_warnings': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download(video_urls)

    return [os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith('.mp3')]

# Constants
SUPPORTED_FORMATS = ('.mp3', '.wav', '.flac', '.aac', '.m4a', '.wma', '.ogg')
TARGET_FREQUENCY = 432.0
DEFAULT_TARGET_SAMPLE_RATE = 48000
DEFAULT_TARGET_BITRATE = "320k"

# 440Hz tuning frequencies
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

tones = {frequency: tone_freq * frequency / 440.0 for frequency in np.arange(424.0, 448.1, 0.1)}

def speed_change(sound, speed=1.0, target_sample_rate=DEFAULT_TARGET_SAMPLE_RATE):
    """Adjust audio speed and resample"""
    # noinspection PyProtectedMember
    altered = sound._spawn(sound.raw_data, overrides={"frame_rate": int(sound.frame_rate * speed)})
    return altered.set_frame_rate(target_sample_rate)

def analyze_tone(song):
    """Detect dominant frequency using FFT analysis"""
    sample_rate = song.frame_rate
    duration = song.duration_seconds

    # Pad audio to 120s if needed
    if duration < 101:
        pad_ms = int((120 - duration) * 1000)
        audio = song + AudioSegment.silent(duration=pad_ms)
    else:
        audio = song

    # Process first 100s of audio
    mono_audio = audio[:100*1000].set_channels(1)
    samples = np.array(mono_audio.get_array_of_samples())
    
    # Perform FFT
    fft_result = np.fft.fft(samples)
    fft_norm = np.abs(fft_result) / len(fft_result)
    
    # Find best matching frequency
    max_sum, max_freq = 0, 0
    for freq in np.arange(424.0, 448.1, 0.1):
        indices = np.round(tones[freq] / (sample_rate / len(samples))).astype(int)
        current_sum = np.sum(fft_norm[indices])
        if current_sum > max_sum:
            max_sum, max_freq = current_sum, freq

    return max_freq

def convert_to_432hz(input_file, output_folder=None, target_format="mp3", 
                    target_bitrate=DEFAULT_TARGET_BITRATE, target_sample_rate=DEFAULT_TARGET_SAMPLE_RATE):
    """Convert single audio file to 432Hz"""
    try:
        if not input_file.lower().endswith(SUPPORTED_FORMATS):
            print(f"Unsupported format: {input_file}")
            return

        output_folder = output_folder or os.path.dirname(input_file)
        output_file = os.path.join(output_folder, f"432hz_{os.path.splitext(os.path.basename(input_file))[0]}.{target_format}")
        
        if os.path.exists(output_file):
            print(f"Skipping existing file: {output_file}")
            return
        
        print(f"Processing: {input_file}")
        song = AudioSegment.from_file(input_file)
        
        detected_tone = analyze_tone(song)
        print(f"Detected base frequency: {detected_tone:.1f}Hz")
        
        speed_ratio = TARGET_FREQUENCY / detected_tone
        converted = speed_change(song, speed_ratio, target_sample_rate)
        
        converted.export(output_file, format=target_format, bitrate=target_bitrate)
        print(f"Saved to: {output_file}")

    except Exception as e:
        print(f"Error processing {input_file}: {str(e)}")

def batch_convert_directory(directory, output_folder=None, target_format="mp3", 
                           target_bitrate=DEFAULT_TARGET_BITRATE, target_sample_rate=DEFAULT_TARGET_SAMPLE_RATE):
    """Batch convert all supported audio files in directory"""
    files = [f for f in os.listdir(directory) 
             if f.lower().endswith(SUPPORTED_FORMATS) and not f.startswith("432hz_")]
    
    print(f"Converting {len(files)} files in {directory}")
    for i, filename in enumerate(files, 1):
        input_path = os.path.join(directory, filename)
        print(f"Processing {i}/{len(files)}: {filename}")
        convert_to_432hz(input_path, output_folder, target_format, target_bitrate, target_sample_rate)


def main():
    parser = argparse.ArgumentParser(
        description="432Hz Audio Converter for YouTube/BitChute/Rumble",
        epilog='''
Examples:
  YouTube conversion:
    %(prog)s https://youtu.be/example --youtube -o ./output/

  BitChute conversion:
    %(prog)s https://bitchute.com/video/abcd1234/ --youtube -f wav

  Rumble conversion:
    %(prog)s https://rumble.com/video-xyz.html --youtube -s 44100

  Batch convert directory:
    %(prog)s ./music_folder/ -o ./converted/ -b 320k

  Multiple URLs:
    %(prog)s "https://youtu.be/1,https://bitchute.com/vid2" --youtube
''',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("input", help="Input file/directory/URL(s)")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-f", "--format", default="mp3",
                        help="Output format (mp3, wav, etc)")
    parser.add_argument("-b", "--bitrate", default=DEFAULT_TARGET_BITRATE,
                        help="Bitrate (default: 192k)")
    parser.add_argument("-s", "--sample-rate", type=int,
                        default=DEFAULT_TARGET_SAMPLE_RATE,
                        help="Sample rate (default: 48000)")
    parser.add_argument("--youtube", action="store_true",
                        help="REQUIRED for URL processing\n"
                             "Supports: YouTube, BitChute, Rumble, and other\n"
                             "platforms compatible with yt-dlp")

    args = parser.parse_args()

    if args.youtube:
        temp_dir = 'temp_audio_downloads'
        os.makedirs(temp_dir, exist_ok=True)

        try:
            urls = args.input.split(',') if ',' in args.input else [args.input]
            downloaded = download_platform_audio(urls, temp_dir)

            for file in downloaded:
                convert_to_432hz(file, args.output, args.format, args.bitrate, args.sample_rate)

        finally:
            # Cleanup temporary files
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

    elif os.path.isfile(args.input):
        convert_to_432hz(args.input, args.output, args.format, args.bitrate, args.sample_rate)
    elif os.path.isdir(args.input):
        batch_convert_directory(args.input, args.output, args.format, args.bitrate, args.sample_rate)
    else:
        print(f"Invalid input: {args.input}")

if __name__ == "__main__":
    main()
