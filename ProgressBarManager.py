# progress_manager.py
import slicer

class ProgressBarManager:
    def __init__(self, layout):
        self.progressBar = None
        self.layout = layout

    def createProgressBar(self, parent, maximum=100):
        if self.progressBar is None:
            self.progressBar = slicer.util.createProgressDialog(parent=parent,windowTitle='Processing...',autoClose=False)
            self.progressBar.setMaximum(maximum)
            self.layout.addRow(self.progressBar)

    def updateProgress(self, text=None):
        if text is not None:
            print(text)
        if self.progressBar is not None:
            self.progressBar.setValue(self.progressBar.value+1)
            slicer.app.processEvents()

    def closeProgressBar(self):
        if self.progressBar is not None:
            self.progressBar.close()
            self.progressBar = None
