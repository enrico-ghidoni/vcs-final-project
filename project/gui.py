import PySimpleGUI as sg
import pipeline as pipeline
import cv2


class GUIPipeline(object):
    def __init__(self):
        sg.theme('DarkBlue')

    def run(self):
        layout = [[sg.Text('Enter run parameters')],
                    [sg.Text('Video'), sg.Input(key='-video-', default_text='/home/enrico/uni/vision-cognitive-systems/final-project/videos/001/GOPR5818.MP4'), sg.FileBrowse()],
                    [sg.Text('Paintings DB'), sg.Input(key='-paintingsdb-', default_text='/home/enrico/uni/vision-cognitive-systems/final-project/data/paintings_db'), sg.FolderBrowse()],
                    [sg.Text('Paintings CSV'), sg.Input(key='-paintingscsv-', default_text='/home/enrico/uni/vision-cognitive-systems/final-project/data/data.csv'), sg.FileBrowse()],
                    [sg.Text('People detection config'), sg.Input(key='-ppldetection-', default_text='/home/enrico/uni/vision-cognitive-systems/final-project/project/people_det_config'), sg.FolderBrowse()],
                    [sg.Text('Output path'), sg.Input(key='-outputpath-', default_text='/home/enrico/uni/vision-cognitive-systems/final-project/out'), sg.FolderBrowse()],
                    [sg.Text('Pick one frame in'), sg.InputText('1', key='-onein-'), sg.Text('frames')],
                    [sg.Submit(), sg.Cancel()]]
        window = sg.Window('Enter parameters', layout)
        event, values = window.read()
        window.close()

        self._pipe = pipeline.Pipeline(
            values['-video-'],
            values['-paintingsdb-'],
            values['-paintingscsv-'],
            values['-outputpath-'],
            values['-ppldetection-'],
            int(values['-onein-'])
        )

        layout = [[sg.Image(filename='', key='-image-')],
                    [sg.StatusBar(text='Stopped', size=(113, 1), key='-status-')]]
        self._window = sg.Window('Running', layout)
        self._pipe.start()
        while self._pipe.next_frame():
            event, values = self._window.read(timeout=15)

            self._update_image()
            self._update_status_bar()

        window.close()

    def _update_image(self):
        image = self._pipe.image_matching_bounding
        image = cv2.resize(image, (1024, 512))
        imgbytes = cv2.imencode('.png', image)[1].tobytes()

        self._window['-image-'].update(data=imgbytes)

    def _update_status_bar(self):
        framen = self._pipe._cur_frame
        tot_bounding = 0
        tot_match = 0
        ppl_found = 0
        if self._pipe.frame_bounding_boxes:
            tot_bounding = len(self._pipe.frame_bounding_boxes)
            tot_match = sum(match is not None for match in self._pipe.frame_ims_matches)
        if self._pipe.frame_ppl_bounding_boxes is not None:
            ppl_found = sum(person is not None for person in self._pipe.frame_ppl_bounding_boxes)
        text = f'Running: frame {framen}, detected {tot_bounding} paintings, found {tot_match} matches, detected {ppl_found} persons'

        self._window['-status-'].update(text)


def console_entry_point():
    runner = GUIPipeline()
    runner.run()

if __name__ == '__main__':
    console_entry_point()
