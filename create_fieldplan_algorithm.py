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
                       QgsProcessingParameterVectorLayer,
                       QgsProcessingParameterBoolean,
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
Builds a field plan, based on given parameters, with functionality to have multiple different blocks (this can be done by giving multiple comma-seperated values in the block details).\n
Field plan parameters are split into two sections - site parameters and block parameters.\n

NOTE: For a visualisation of these parameters, go to: <a href="https://github.com/kieranatkins/qgis-fieldplan/blob/main/site_info.png"> site parameters </a> and <a href="https://github.com/kieranatkins/qgis-fieldplan/blob/main/block_info.png"> block parameters </a>\n
----------\n
Site info:\n
----------\n
Origin - The initial coordinate, this will be the starting point from where the field plan will be built, this is where the plot with id=0 will be and should be along the field edge.\n
Bearing point - A second point used to establish the bearing of the field edge, where the first row will be built.\n
Direction - The direction along the bearing to build the field plan (left or right, from origin to destination).\n
Margin - Empty space from the origin before building the field plan, this accepts either a single value for margin, or two comma-separated values for x (row dimension) and y (column dimension) independently .\n
Blockgap - Gap between blocks.\n
----------\n
Block info:\n
----------\n
Rows - The number of rows within the block (the number parallel to the bearing).\n
Columns - The number of columns within the block (the number perpendicular to the bearing).\n
Plots per board - The number of plots within a board (these will be built in rows parallel to the bearing).\n
Dimension 1 (Board length) - The size of a board in the bearing's dimension (i.e. the row's dimension)\n
Dimension 2 (Board width) - The size of the board in the dimension perpendicular to the bearing (i.e. the column's dimension) \n
Dimension 3 (Row gap)- The spacing/gap from one board to the next within a row.\n
Dimension 4 (Plot gap)- The spacing/gap between plots in a board.\n
Dimension 5 (Column gap)- The spacing/gap between one board to the next within a column.\n

"""
LAYER_NAME = "Field plan"
BEARING_LAYER_NAME = "Field plan bearing"
BLOCK_PARAMETERS = ['COLS', 'ROWS', 'PPB', 'DIM1', 'DIM2', 'DIM3', 'DIM4', 'DIM5']

# LLM function
def _find_line_point_distance(line_x1, line_y1, line_x2, line_y2, point_x, point_y):
    """
    Calculate the shortest distance between a point and a line segment, and find the closest point on the line.
    
    Args:
        line_x1, line_y1: Coordinates of the first endpoint of the line
        line_x2, line_y2: Coordinates of the second endpoint of the line  
        point_x, point_y: Coordinates of the point
    
    Returns:
        tuple: (distance, closest_x, closest_y) where:
            - distance: shortest distance between point and line
            - closest_x, closest_y: coordinates of the closest point on the line
    """
    # Calculate line vector components
    line_dx = line_x2 - line_x1
    line_dy = line_y2 - line_y1
    
    # Calculate line length squared
    line_length_squared = line_dx ** 2 + line_dy ** 2
    
    # Handle case where line points are identical
    if line_length_squared == 0:
        closest_x, closest_y = line_x1, line_y1
        distance = math.sqrt((line_x1 - point_x) ** 2 + (line_y1 - point_y) ** 2)
        return distance, closest_x, closest_y
    
    # Calculate projection parameter t
    # t represents how far along the line the closest point is (0 = at line_x1, 1 = at line_x2)
    t = ((point_x - line_x1) * line_dx + (point_y - line_y1) * line_dy) / line_length_squared
    
    # Clamp t to [0, 1] to ensure the closest point is on the line segment
    t = max(0, min(1, t))
    
    # Calculate the closest point on the line
    closest_x = line_x1 + t * line_dx
    closest_y = line_y1 + t * line_dy
    
    # Calculate the distance
    distance = math.sqrt((closest_x - point_x) ** 2 + (closest_y - point_y) ** 2)
    
    return distance, closest_x, closest_y

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
                'BEARINGPOINT',
                self.tr('Bearing point'),
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                'SNAPTOSHAPE',
                self.tr('Snap to shape'),
                optional=True,
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
            QgsProcessingParameterString(
                'MARGIN',
                self.tr('Margin'),
                defaultValue="0"
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
                self.tr('Dimension 1 (Board length)'),
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                'DIM2',
                self.tr('Dimension 2 (Board width)'),
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
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                'FIELDPLANBEARING',
                self.tr('Output field plan bearing')
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                'RETURNBEARING',
                self.tr('Return bearing?'),
                optional=False,
                defaultValue=True
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
        
        margin = self.parameterAsString(parameters, 'MARGIN', context).split(',')
        if len(margin) > 2:
            return False, "Margin parameter may contain either one value, to represent margin in both X and Y direction, or two comma-seperated values representing margin in X, Y directions"
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
        dest = self.parameterAsPoint(parameters, 'BEARINGPOINT', context)
        snap_to_shape = self.parameterAsVectorLayer(parameters, 'SNAPTOSHAPE', context)
        return_bearing = self.parameterAsBoolean(parameters, 'RETURNBEARING', context)
        left = self.parameterAsInt(parameters, 'DIRECTION', context) == 0
        margin = [float(x) for x in self.parameterAsString(parameters, 'MARGIN', context).split(',')]
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

        if snap_to_shape is None:
            origin_x = origin.x()
            origin_y = origin.y()
            dest_x = dest.x()
            dest_y = dest.y()
        else:
            snapped_points = []
            for x, y in [(origin.x(), origin.y()), (dest.x(), dest.y())]:
                # find closest line
                closest_x = None
                closest_y = None
                closest_line_distance = math.inf

                # not sure why shapes are nested like this, but this is probably the safest way to deal with it?
                for feat in snap_to_shape.getFeatures():
                    geom = feat.geometry().as_numpy()
                    for shape1 in geom:
                        for shape2 in shape1:
                            for p1, p2 in zip(shape2, shape2[1:]):
                                dist, _x, _y = _find_line_point_distance(p1[0], p1[1], p2[0], p2[1], x, y)
                                if dist < closest_line_distance:
                                    closest_x, closest_y = _x, _y
                                    closest_line_distance = dist
                
                snapped_points.append([closest_x, closest_y])

            (origin_x, origin_y), (dest_x, dest_y) = snapped_points

            feedback.pushInfo(f'Updated origin from {origin.x():.2f}, {origin.y():.2f} to {origin_x:.2f}, {origin_y:.2f}')
            feedback.pushInfo(f'Updated dest from {dest.x():.2f}, {dest.y():.2f} to {dest_x:.2f}, {dest_y:.2f}')

        if len(margin) == 2:
            margin_x = margin[0]
            margin_y = margin[1]
        else:
            margin_x = margin[0]
            margin_y = margin[0]

        # calculate bearing
        dx = dest_x - origin_x
        dy = dest_y - origin_y
        bearing = math.degrees(math.atan2(dx, dy)) % 360

        feedback.pushInfo(f'Calculated bearing to be: {bearing}')

        block_origin_x = margin_x
        block_origin_y = margin_y

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
                        bl_x += col * ((ppb * two) + ((ppb-1) * four) + five)
                        bl_x += p * (two + four)

                        bl_y = block_origin_y
                        bl_y += row * (one + three)    

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

                            x = origin_x + rotated_x
                            y = origin_y - rotated_y

                            polygon.append((x, y))
                        
                        polygons.append(polygon)
                        row_ids.append(int(row+1))
                        column_ids.append(int(col+1))
                        plot_ids.append(int(p+1))
                        block_ids.append(int(b+1))

            block_origin_y += ((cols * one) + ((cols - 1) * three) + block_gap)     

        for i, (polygon, block, row, column, plot) in enumerate(zip(polygons, block_ids, row_ids, column_ids, plot_ids)):
            polygon = [QgsPointXY(x,y) for x,y in polygon]
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPolygonXY([polygon]))  #xmin, ymin, xmax, ymax
            feat.setAttributes([i+1, block, row, column, plot])
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

        ret = {'FIELDPLAN':layer}

        if return_bearing:
            crs = context.project().crs().authid()
            bearing_layer = QgsVectorLayer(f"Linestring?crs={crs}", BEARING_LAYER_NAME, "memory")
            bearing_layer.startEditing()

            bearing_provider = bearing_layer.dataProvider()
            bearing_fields = QgsFields()
            bearing_fields.append(QgsField("bearing", QMetaType.Double))
            bearing_provider.addAttributes(bearing_fields)
            bearing_layer.updateFields()
            feat = QgsFeature()
            feat.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(origin_x, origin_y), QgsPointXY(dest_x, dest_y)]))
            feat.setAttributes([bearing])
            bearing_provider.addFeature(feat)
            bearing_layer.commitChanges()
            bearing_layer.updateExtents()
            context.temporaryLayerStore().addMapLayer(bearing_layer)
            context.addLayerToLoadOnCompletion(
                bearing_layer.id(),
                QgsProcessingContext.LayerDetails(
                    BEARING_LAYER_NAME,
                    context.project(),
                    'FIELDPLANBEARING'
                )
            )
            ret['FIELDPLANBEARING'] = bearing_layer

        feedback.setProgress(100)        
        return ret
