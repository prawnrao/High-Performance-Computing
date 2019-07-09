class Grid(object):
    """This class implements access to triangular grids."""
    def __init__(self, vertices, elements):
        """
        Initialize a grid.

        This routine takes an Nx2 Numpy array of N vertices
        and a Mx3 Numpy array with corresponding M elements.
        """
        import numpy as np

        if not (isinstance(vertices, np.ndarray) and
                isinstance(elements, np.ndarray)):
            raise ValueError("The input data must be of type numpy.ndarray.")
        self.__vertices = vertices
        self.__elements = elements

        # Some protection against modifying the grid data externally
        self.__vertices.setflags(write=False)
        self.__elements.setflags(write=False)

    @classmethod
    def from_vtk_file(cls, filename):
        """Create a grid from a given vtk file."""

        # Insert code that reads a grid from a vtk file.
        # For this you should look up the VtkUnstructuredGridReader class.
        # Make sure that you only import triangular elements (check the vtk
        # cell type).
        # VTK only knows vertices in 3 dimensions. Simply ignore the
        # z-coordinate.

        import os.path
        import vtk
        import numpy as np
        if not os.path.isfile(filename):
            raise ValueError("File does not exist.")

        # Reads the vtk file and stores the result in the output variable
        reader = vtk.vtkUnstructuredGridReader()
        reader.SetFileName(filename)
        reader.Update()
        output = reader.GetOutput()

        # Extracts the x and y coordinates of the points
        # and creates an array of points.
        points = np.array(output.GetPoints().GetData())
        points = points[:, :2]
        numberOfCells = output.GetNumberOfCells()

        # Extracts the point Ids from each cell and creates an
        # array of elements containing point Ids.
        i = 0
        first = True
        while i < numberOfCells:
            cell = output.GetCell(i)
            if cell.GetCellType() == 5:
                numpyCell = np.array([cell.GetPointId(0),
                                     cell.GetPointId(1),
                                     cell.GetPointId(2)])
                if first:
                    elements = numpyCell
                    first = False
                else:
                    elements = np.vstack([elements, numpyCell])
            i += 1

        return cls(points, elements)

    @property
    def number_of_vertices(self):
        """Return the number of vertices."""
        return self.__vertices.shape[0]

    @property
    def number_of_elements(self):
        """Return the number of elements."""
        return self.__elements.shape[0]

    @property
    def vertices(self):
        """Return the vertices."""
        return self.__vertices

    @property
    def elements(self):
        """Return the elements."""
        return self.__elements

    def get_corners(self, element_id):
        """Return the 3x2 matrix of corners associated with an element."""
        import numpy as np
        element = self.__elements[element_id]
        first = True
        for point_id in element:
            if first:
                corners = self.__vertices[point_id]
                first = False
            else:
                corners = np.vstack([corners, self.__vertices[point_id]])
        return corners

    def get_jacobian(self, element_id):
        """Return the jacobian associated with a given element id."""
        import numpy as np
        corners = self.get_corners(element_id)
        jacobian = np.vstack([corners[1] - corners[0],
                              corners[2] - corners[0]]).transpose()
        return jacobian

    def export_to_vtk(self, fname, point_data=None):
        """Export grid to a vtk file. Optionally also export point data."""
        from vtk import vtkUnstructuredGrid, vtkPointData, vtkDoubleArray, \
            vtkPoints, vtkUnstructuredGridWriter, VTK_TRIANGLE

        grid = vtkUnstructuredGrid()

        if point_data is not None:

            data = grid.GetPointData()
            scalar_data = vtkDoubleArray()
            scalar_data.SetNumberOfValues(len(point_data))
            for index, value in enumerate(point_data):
                scalar_data.SetValue(index, value)
            data.SetScalars(scalar_data)

        points = vtkPoints()
        points.SetNumberOfPoints(self.number_of_vertices)
        for index in range(self.number_of_vertices):
            points.InsertPoint(
                index,
                (self.vertices[index, 0], self.vertices[index, 1], 0))

        grid.SetPoints(points)

        for index in range(self.number_of_elements):
            grid.InsertNextCell(
                VTK_TRIANGLE, 3,
                [self.elements[index, 0], self.elements[index, 1],
                 self.elements[index, 2]]
            )

        writer = vtkUnstructuredGridWriter()
        writer.SetFileName(fname)
        writer.SetInputData(grid)
        writer.Write()

        return grid
