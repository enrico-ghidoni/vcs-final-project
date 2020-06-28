import PySimpleGUI as sg
import project.pipeline as pipeline
import cv2


class GUIPipeline(object):
    def __init__(self):
        sg.theme('DarkBlue')

    def run(self):
        layout = [[sg.Text('Enter run parameters')],
                    [sg.Text('Video'), sg.Input(key='-video-', default_text='/home/enrico/uni/vision-cognitive-systems/final-project/videos/001/GOPR5818.MP4'), sg.FileBrowse()],
                    [sg.Text('Paintings DB'), sg.Input(key='-paintingsdb-', default_text='/home/enrico/uni/vision-cognitive-systems/final-project/data/paintings_db/'), sg.FolderBrowse()],
                    [sg.Text('Paintings CSV'), sg.Input(key='-paintingscsv-', default_text='/home/enrico/uni/vision-cognitive-systems/final-project/data/data.csv'), sg.FileBrowse()],
                    [sg.Text('Output path'), sg.Input(key='-outputpath-', default_text='/home/enrico/uni/vision-cognitive-systems/final-project/out'), sg.FolderBrowse()],
                    [sg.Text('Pick one frame in'), sg.InputText('1', key='-onein-'), sg.Text('frames')],
                    [sg.Submit(), sg.Cancel()]]
        window = sg.Window('Enter parameters', layout)
        event, values = window.read()
        window.close()

        self._pipe = pipeline.Pipeline(values['-video-'], values['-paintingsdb-'], values['-paintingscsv-'], values['-outputpath-'], int(values['-onein-']), debug = False, silent = True)

        layout = [[sg.Image(filename='', key='-image-')],
                    [sg.StatusBar(text='Stopped', auto_size_text=True, key='-status-')]]
        window = sg.Window('Running', layout)
        self._pipe.start()
        while self._pipe.next_frame():
            event, values = window.read(timeout=15)
            image = self._pipe.image_matching_bounding
            imgbytes = cv2.imencode('.png', self._pipe.image_matching_bounding)[1].tobytes()
            window['-image-'].update(data=imgbytes)
            window['-status-'].update(f'Running: frame {self._pipe._cur_frame}')

        window.close()


def console_entry_point():
    runner = GUIPipeline()
    runner.run()

if __name__ == '__main__':
    console_entry_point()
