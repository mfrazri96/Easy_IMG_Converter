class FileQueue:
    def __init__(self):
        self.selected_files = []
        self.path_to_row = {}
        self.row_to_path = {}

    def contains(self, path: str) -> bool:
        return path in self.selected_files

    def add(self, path: str, row_id: str) -> None:
        self.selected_files.append(path)
        self.path_to_row[path] = row_id
        self.row_to_path[row_id] = path

    def remove_by_row(self, row_id: str) -> None:
        path = self.row_to_path.get(row_id)
        if path and path in self.selected_files:
            self.selected_files.remove(path)
            self.path_to_row.pop(path, None)
        self.row_to_path.pop(row_id, None)

    def clear(self) -> None:
        self.selected_files = []
        self.path_to_row = {}
        self.row_to_path = {}

    def row_for_path(self, path: str):
        return self.path_to_row.get(path)

    def path_for_row(self, row_id: str):
        return self.row_to_path.get(row_id)

    def __len__(self):
        return len(self.selected_files)
