from collections import defaultdict
import math

from qgis.PyQt.QtCore import QCoreApplication
import qgis
from qgis.core import (QgsProcessing,
                       QgsProcessingAlgorithm,
                       QgsProcessingException,
                       QgsProcessingParameterPoint,
                       QgsProcessingParameterNumber,
                       QgsProcessingParameterEnum,
                       QgsProcessingOutputVectorLayer,
                       QgsProcessingParameterDefinition,
                       QgsProcessingParameterString,
                       QgsProcessingContext)
from qgis.core import QgsMessageLog
from qgis import processing
from qgis.core import QgsVectorLayer, QgsField, QgsFields, QgsFeature, QgsGeometry, QgsProject, QgsPointXY
from qgis.PyQt.QtCore import QVariant, QMetaType

from qgis.utils import iface

MAX_BLOCKS = 32
INFO = """
Buids a field plan, based on given details about the plot, with functionality to have multiple different blocks (this can be done by giving multiple comma-seperated values in the block details).\n
Field plan parameters are split into two sections - plot info and block info.\n 
----------\n
Plot info:\n
----------\n
Origin - The initial coordinate, this will be the starting point from where the field plan will be built, this is where the 0-id field
will be and should be along the field edge\n
Destination - A coordinate along the field edge to create a bearing from which the field plan will be built.\n
Direction - The direction along the bearing to build the field plan (left or right, from origin to destination).\n
Margin - Empty space from the origin before building the field plan.\n
Blockgap - Margin between blocks.\n
----------\n
Block info:\n
----------\n
Rows - The number of rows within the block (the number parallel to the bearing)\n
Columns - The number of columns within the block (the number perpendicular to the bearing)\n
Plots per board - The number of plots within a board (these will be built in rows parallel to the bearing)\n
Dimension 1 (Board width) - The width of a plot within a board (or simply with of the board)\n
Dimension 2 (Board height) - The height of a plot within a board \n
Dimension 3 (Row gap)- The spacing/gap from one board to the next within a row\n
Dimension 4 (Plot gap)- The spacing/gap between plots in a board\n
Dimension 5 (Column gap)- The spacing/gap between one board to the next within a column\n

NOTE: For a visualisation of these dimensions, go to: <LINK>

"""
LAYER_NAME = "Field plan"
BLOCK_PARAMETERS = ['COLS', 'ROWS', 'PPB', 'DIM1', 'DIM2', 'DIM3', 'DIM4', 'DIM5']

class CreateFieldPlan(QgsProcessingAlgorithm):
    """
    This is an example algorithm that takes a vector layer,
    creates some new layers and returns some results.
    """

    def tr(self, string):
        """
        Returns a translatable string with the self.tr() function.
        """
        return QCoreApplication.translate('Processing', string)

    def createInstance(self):
        # Must return a new copy of your algorithm.
        return CreateFieldPlan()

    def name(self):
        """
        Returns the unique algorithm name.
        """
        return 'field_plan_create'

    def displayName(self):
        """
        Returns the translated algorithm name.
        """
        return self.tr('Create new field plan')

    def group(self):
        """
        Returns the name of the group this algorithm belongs to.
        """
        return self.tr('Field plan')

    def groupId(self):
        """
        Returns the unique ID of the group this algorithm belongs
        to.
        """
        return 'fieldplan'

    def shortHelpString(self):
        """
        Returns a localised short help string for the algorithm.
        """
        return self.tr(INFO)

    def initAlgorithm(self, config=None):
        """
        Here we define the inputs and outputs of the algorithm.
        """
        self.addParameter(
            QgsProcessingParameterPoint(
                'ORIGIN',
                self.tr('Origin'),
            )
        )
        self.addParameter(
            QgsProcessingParameterPoint(
                'DESTINATION',
                self.tr('Destination'),
            )
        )
        self.addParameter(
            QgsProcessingParameterEnum(
                'DIRECTION',
                self.tr('Direction'),
                options=['Left', 'Right']
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                'MARGIN',
                self.tr('Margin'),
                type=qgis.core.Qgis.ProcessingNumberParameterType.Double,
                defaultValue=0.0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                'BLOCKGAP',
                self.tr('Block gap'),
                type=qgis.core.Qgis.ProcessingNumberParameterType.Double,
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                'BLOCKNUMBER',
                self.tr('Block number'),
                type=qgis.core.Qgis.ProcessingNumberParameterType.Integer,
                defaultValue=0,
                minValue=1,
                maxValue=MAX_BLOCKS
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                'ROWS',
                self.tr('Rows in a block'),
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                'COLS',
                self.tr('Columns in a block'),
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                'PPB',
                self.tr('Plots per board'),
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                'DIM1',
                self.tr('Dimension 1 (Board width)'),
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                'DIM2',
                self.tr('Dimension 2 (Board height)'),
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                'DIM3',
                self.tr('Dimension 3 (Row gap)'),
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                'DIM4',
                self.tr('Dimension 4 (Plot gap)'),
            )
        )

        self.addParameter(
            QgsProcessingParameterString(
                'DIM5',
                self.tr('Dimension 5 (Column gap)'),
            )
        )
        
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                'FIELDPLAN',
                self.tr('Output field plan')
            )
        )


    def checkParameterValues(self, parameters, context):
        # Check CRS
        crs = context.project().crs()
        if crs.isGeographic():
            return False, "The system requires the CRS be geographic (e.g. UTM projection), as the script calculates shapes in metres"
        # Check parameters
        bn = self.parameterAsInt(parameters, 'BLOCKNUMBER', context)
        for p in BLOCK_PARAMETERS:
            s = self.parameterAsString(parameters, p, context)
            block_params = s.split(',')
            if len(block_params) != bn:
                return False, "At least one of the block parameters (i.e. rows, columns, plots per board, dimensions) contains the wrong number of values for the number of blocks (values should be comma-seperated)"
        
        return True, ''


    def processAlgorithm(self, parameters, context, feedback):
        """
        Here is where the processing itself takes place.
        """
        # Create empty vector based on current project CRS
        crs = context.project().crs().authid()
        layer = QgsVectorLayer(f"Polygon?crs={crs}", LAYER_NAME, "memory")
        layer.startEditing()
    
        # Setup fields
        provider = layer.dataProvider()
        fields = QgsFields()
        fields.append(QgsField("id", QMetaType.Int))
        fields.append(QgsField("block", QMetaType.Int))
        fields.append(QgsField("row", QMetaType.Int))
        fields.append(QgsField("col", QMetaType.Int))
        fields.append(QgsField("plot", QMetaType.Int))
        provider.addAttributes(fields)
        layer.updateFields()

        # Setup parameters
        origin = self.parameterAsPoint(parameters, 'ORIGIN', context)
        dest = self.parameterAsPoint(parameters, 'DESTINATION', context)
        left = self.parameterAsInt(parameters, 'DIRECTION', context) == 0
        margin = self.parameterAsDouble(parameters, 'MARGIN', context)
        block_gap = self.parameterAsDouble(parameters, 'BLOCKGAP', context)
        block_number = self.parameterAsInt(parameters, 'BLOCKNUMBER', context)
        col_info = [int(x) for x in self.parameterAsString(parameters, 'COLS', context).split(',')]
        row_info = [int(x) for x in self.parameterAsString(parameters, 'ROWS', context).split(',')]
        plots_per_board = [int(x) for x in self.parameterAsString(parameters, 'PPB', context).split(',')]
        dim1 = [float(x) for x in self.parameterAsString(parameters, 'DIM1', context).split(',')]
        dim2 = [float(x) for x in self.parameterAsString(parameters, 'DIM2', context).split(',')]
        dim3 = [float(x) for x in self.parameterAsString(parameters, 'DIM3', context).split(',')]
        dim4 = [float(x) for x in self.parameterAsString(parameters, 'DIM4', context).split(',')]
        dim5 = [float(x) for x in self.parameterAsString(parameters, 'DIM5', context).split(',')]

        # calculate bearing
        dx = dest.x() - origin.x()
        dy = dest.y() - origin.y()
        bearing = math.degrees(math.atan2(dx, dy)) % 360

        feedback.pushInfo(f'Calculated bearing to be: {bearing}')

        block_origin_x = margin
        block_origin_y = margin

        polygons = []
        block_ids = []
        column_ids = []
        row_ids = []
        plot_ids = []

        # Iterate over each block
        block_params = zip(col_info, row_info, plots_per_board, dim1, dim2, dim3, dim4, dim5)
        for b, (cols, rows, ppb, one, two, three, four, five) in enumerate(block_params):
            for row in range(rows):
                for col in range(cols):
                    for p in range(ppb):
                        # Bottom left                                                       
                        bl_x = block_origin_x     
                        bl_x += row * ((ppb * two) + ((ppb-1) * four) + five)
                        bl_x += p * (two + four)

                        bl_y = block_origin_y
                        bl_y += col * (one + three)    

                        # Bottom right                                                  
                        br_x = bl_x + two
                        br_y = bl_y                                                                                             

                        # Top left
                        tl_x = bl_x
                        tl_y = bl_y + one

                        # Top right
                        tr_x = bl_x + two
                        tr_y = bl_y + one

                        # Must be in clockwise order
                        bearing_radians = math.radians(bearing-180)
                        polygon = []
                        for x, y in [(bl_x, bl_y), (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y)]:
                            x = x if left else -x
                            rotated_x = x * math.cos(bearing_radians) - y * math.sin(bearing_radians)
                            rotated_y = x * math.sin(bearing_radians) + y * math.cos(bearing_radians)

                            x = origin.x() + rotated_x
                            y = origin.y() - rotated_y

                            polygon.append((x, y))
                        
                        polygons.append(polygon)
                        row_ids.append(int(row))
                        column_ids.append(int(col))
                        plot_ids.append(int(p))
                        block_ids.append(int(b))

            block_origin_y += ((cols * one) + ((cols - 1) * three) + block_gap)     

        # error is here
        for i, (polygon, block, row, column, plot) in enumerate(zip(polygons, block_ids, row_ids, column_ids, plot_ids)):
            polygon = [QgsPointXY(x,y) for x,y in polygon]
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPolygonXY([polygon]))  #xmin, ymin, xmax, ymax
            feat.setAttributes([i, block, row, column, plot])
            provider.addFeature(feat)
        
        feedback.pushInfo(f'Created {i+1} polygons.')

        layer.commitChanges()
        layer.updateExtents()
        context.temporaryLayerStore().addMapLayer(layer)
        context.addLayerToLoadOnCompletion(
            layer.id(),
            QgsProcessingContext.LayerDetails(
                LAYER_NAME,
                context.project(),
                'FIELDPLAN'
            )
        )

        feedback.setProgress(100)        
        return {'FIELDPLAN':layer}
