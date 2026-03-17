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
Builds a field plan, based on given parameters, with functionality to have multiple blocks.\n
Field plan parameters are split into three sections - site parameters and block parameters.\n
Note: Asterisks (*) denotes places where comma-separated values can be used.

The algorithm denotes two dimensions: The X-axis, perpendicular the bearing line and the Y-axis, parallel to the bearing line

NOTE: For a visualisation of these parameters, go to: <a href="https://github.com/kieranatkins/qgis-fieldplan/blob/main/site_info.png"> site parameters </a> and <a href="https://github.com/kieranatkins/qgis-fieldplan/blob/main/block_info.png"> block parameters </a>\n
--------------------------\n
Site parameters:\n
--------------------------\n
Origin - The initial coordinate, this will be the starting point from where the field plan will be built, this is where the plot with id=0 will be and should be along the field edge.\n
Bearing point - A second point used to establish the bearing of the field edge, where the first column will be built.\n
Snap to shape - An optional parameter that snaps the origin and bearing points to the closest edge of a given layer
Direction - The direction along the bearing to build the field plan (left or right, from origin to destination).\n
Margin - Empty space from the origin before building the field plan, this accepts either a single value for margin, or two comma-separated values for X&Y independently .\n
Block number - Number of blocks
Block gap - The spacing/gap from one block to a next.\n

--------------------------\n
Variable block parameters:\n
--------------------------\n
Note: Either a single value or multiple comma-seperated values can be placed in these input fields representing each block
Rows - The number of rows within the block (the number parallel to the bearing).\n
Columns - The number of columns within the block (the number perpendicular to the bearing).\n
Plot length - The length of a plot in the y dimension
Alley - The spacing/gap from one row to a next within a column.\n

--------------------------\n
Fixed block parameters:\n
--------------------------\n
Plots per column - The number of plots within a single column
Dimension 1 - The plot width
Dimension 2 - The plot gap
Dimension 3 - The column gap
Dimension 4 - The column margin - An optional spacing parameter to allow for dead space between plot and vehicle wheel width

"""
LAYER_NAME = "Field plan"
BEARING_LAYER_NAME = "Field plan bearing"
GUIDANCE_LINE_LAYER_NAME = "Guidance lines"
ALLEY_LINE_LAYER_NAME = "Alley lines"
BLOCK_BOUNDARY_LAYER_NAME = 'Block boundaries'
BOUNDARY_LAYER_NAME = "Field plan block boundaries"
VARIABLE_PARAMETERS = ['COLS', 'ROWS', 'PLOTLENGTH', 'ALLEY']
FIELD_GUIDANCE_LINES_PARAMETERS = ['DIM1' ,'DIM2', 'DIM3', 'DIM4']

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

def _create_field_plan(origin, margin, bearing, left, block_gap, block_rows, block_cols, block_plot_length, block_alley, ppc, one, two, three, four):
    origin_x, origin_y = origin
    block_origin_x, block_origin_y = margin

    polygons = []
    block_ids = []
    column_ids = []
    row_ids = []
    plot_ids = []

    guidance_lines_coords = []
    guidance_lines_col = []

    alley_lines_coords = []
    alley_lines_block = []
    alley_lines_row = []

    boundary_lines_coords = []
    boundary_lines_block = []

    block_boundaries = []

    # Iterate over each block
    for b, (cols, rows, pl, a) in enumerate(zip(block_cols, block_rows, block_plot_length, block_alley)):
        block_boundaries.append(block_origin_y)
        for row in range(rows):
            for col in range(cols):
                for p in range(ppc):
                    # Bottom left                                                       
                    bl_x = block_origin_x + four
                    bl_x += col * ((ppc * one) + ((ppc-1) * two) + three + four*2)
                    bl_x += p * (one + two)

                    bl_y = block_origin_y
                    bl_y += row * (pl + a)    

                    # Bottom right                                                  
                    br_x = bl_x + one
                    br_y = bl_y                                                                                             

                    # Top left
                    tl_x = bl_x
                    tl_y = bl_y + pl

                    # Top right
                    tr_x = bl_x + one
                    tr_y = bl_y + pl

                    # Must be in clockwise order
                    polygon = []
                    for x, y in [(bl_x, bl_y), (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y)]:
                        x = x if left else -x
                        rotated_x = x * math.cos(bearing) - y * math.sin(bearing)
                        rotated_y = x * math.sin(bearing) + y * math.cos(bearing)

                        x = origin_x + rotated_x
                        y = origin_y - rotated_y

                        polygon.append((x, y))
                    
                    polygons.append(polygon)
                    row_ids.append(int(row+1))
                    column_ids.append(int(col+1))
                    plot_ids.append(int(p+1))
                    block_ids.append(int(b+1))

            # alley lines - don't do the last row
            if row < rows-1:
                x1 = block_origin_x + four
                x2 = block_origin_x + four + cols * ((ppc * one) + ((ppc-1) * two)) + ((cols-1) * (three + (four*2)))
                y = block_origin_y + (row * (pl + a)) + pl + a/2

                # rotate
                al = [[x1, y], [x2, y]]
                al_rotated = []
                for x, y in al:
                    x = x if left else -x

                    rotated_x = x * math.cos(bearing) - y * math.sin(bearing)
                    rotated_y = x * math.sin(bearing) + y * math.cos(bearing)

                    x = origin_x + rotated_x
                    y = origin_y - rotated_y

                    al_rotated.extend([x, y])

                alley_lines_coords.append(al_rotated)
                alley_lines_row.append(row)
                alley_lines_block.append(b)
        
        low_x1 = block_origin_x + four
        low_x2 = block_origin_x + four + cols * ((ppc * one) + ((ppc-1) * two)) + ((cols-1) * (three + (four*2)))
        low_y = block_origin_y

        hi_x1 = block_origin_x + four
        hi_x2 = block_origin_x + four + cols * ((ppc * one) + ((ppc-1) * two)) + ((cols-1) * (three + (four*2)))
        hi_y = block_origin_y + (rows * pl) + ((rows - 1) * a)

        for x1, y1, x2, y2 in [[low_x1, low_y, low_x2, low_y], [hi_x1, hi_y, hi_x2, hi_y]]:
            rotated = []
            for x, y in [[x1, y1], [x2, y2]]:
                x = x if left else -x

                rotated_x = x * math.cos(bearing) - y * math.sin(bearing)
                rotated_y = x * math.sin(bearing) + y * math.cos(bearing)

                x = origin_x + rotated_x
                y = origin_y - rotated_y

                rotated.extend([x, y])

                boundary_lines_coords.append(rotated)
                boundary_lines_block.append(b)

        block_origin_y += ((rows * pl) + ((rows - 1) * a) + block_gap)

    # calculate column IDs for each block
    col_block_ids = [list(range(c)) for c in block_cols]
    for col in range(max(block_cols)):
        # calculate x coord
        boundary1 = margin[0] + four + col * ((one * ppc) + (two * (ppc-1)) + three + (four*2))
        boundary2 = boundary1 + (one * ppc) + (two * (ppc-1))
        x = (boundary1 + boundary2) / 2

        # find which block IDs has the current column
        block_ids_col = [i for i, c in enumerate(col_block_ids) if col in c]
        block1 = min(block_ids_col)
        block2 = max(block_ids_col)

        y1 = block_boundaries[block1] - 1
        y2 = block_boundaries[block2] + (block_rows[block2] * block_plot_length[block2]) + ((block_rows[block2] - 1) * block_alley[block2]) + 1
        gl = [[x, y1], [x, y2]]

        # rotate
        gl_rotated = []
        for x, y in gl:
            x = x if left else -x

            rotated_x = x * math.cos(bearing) - y * math.sin(bearing)
            rotated_y = x * math.sin(bearing) + y * math.cos(bearing)

            x = origin_x + rotated_x
            y = origin_y - rotated_y

            gl_rotated.extend([x, y])

        guidance_lines_coords.append(gl_rotated)
        guidance_lines_col.append(col)

    return polygons, block_ids, column_ids, row_ids, plot_ids, guidance_lines_coords, guidance_lines_col, alley_lines_coords, alley_lines_row, alley_lines_block, boundary_lines_coords, boundary_lines_block


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
                self.tr('Margin*'),
                defaultValue="0"
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
            QgsProcessingParameterNumber(
                'BLOCKGAP',
                self.tr('Block gap'),
                type=qgis.core.Qgis.ProcessingNumberParameterType.Double,
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                'ROWS',
                self.tr('Rows in a block*'),
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                'COLS',
                self.tr('Columns in a block*'),
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                'PLOTLENGTH',
                self.tr('Plot length*'),
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                'ALLEY',
                self.tr('Alley*'),
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                'PPC',
                self.tr('Plots per column'),
                type=qgis.core.Qgis.ProcessingNumberParameterType.Integer,
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'DIM1',
                self.tr('Dimension 1 (Plot width)'),
                type=qgis.core.Qgis.ProcessingNumberParameterType.Double,
                defaultValue=0
            )
        )

        self.addParameter(
            QgsProcessingParameterNumber(
                'DIM2',
                self.tr('Dimension 2 (Plot gap)'),
                type=qgis.core.Qgis.ProcessingNumberParameterType.Double,
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                'DIM3',
                self.tr('Dimension 3 (Column gap)'),
                type=qgis.core.Qgis.ProcessingNumberParameterType.Double,
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                'DIM4',
                self.tr('Dimension 4 (Column margin)'),
                type=qgis.core.Qgis.ProcessingNumberParameterType.Double,
                defaultValue=0
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                'RETURNBEARING',
                self.tr('Return bearing'),
                optional=False,
                defaultValue=False
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                'RETURNGUIDANCELINES',
                self.tr('Calculate guidance lines'),
                optional=False,
                defaultValue=False
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                'RETURNALLEYLINES',
                self.tr('Calculate alley lines'),
                optional=False,
                defaultValue=False
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                'RETURNBLOCKBOUNDARIES',
                self.tr('Calculate block boundaries'),
                optional=False,
                defaultValue=False
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
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                'FIELDPLANGUIDANCELINES',
                self.tr('Output block guidance lines')
            )
        )
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                'FIELDPLANALLEYLINES',
                self.tr('Output alley lines')
            )
        )
        self.addOutput(
            QgsProcessingOutputVectorLayer(
                'FIELDPLANBLOCKBOUNDARIES',
                self.tr('Output block edge boundary lines')
            )
        )

    def checkParameterValues(self, parameters, context):
        # Check CRS
        crs = context.project().crs()
        if crs.isGeographic():
            return False, "The system requires the CRS to be geographic (e.g. UTM projection), as the script calculates shapes in metres"
        # Check parameters
        bn = self.parameterAsInt(parameters, 'BLOCKNUMBER', context)

        for p in VARIABLE_PARAMETERS:
            s = self.parameterAsString(parameters, p, context)
            block_params = s.split(',')
            if not (len(block_params) == bn or len(block_params) == 1):
                return False, f"The parameter {p} contains the wrong number of values for the number of blocks ({bn}). Values can be single, or comma-seperated for each block"            

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
        return_guidance_lines = self.parameterAsBoolean(parameters, 'RETURNGUIDANCELINES', context)
        return_alley_lines = self.parameterAsBoolean(parameters, 'RETURNALLEYLINES', context)
        return_block_boundaries = self.parameterAsBoolean(parameters, 'RETURNBLOCKBOUNDARIES', context)
        left = self.parameterAsInt(parameters, 'DIRECTION', context) == 0
        margin = [float(x) for x in self.parameterAsString(parameters, 'MARGIN', context).split(',')]
        block_gap = self.parameterAsDouble(parameters, 'BLOCKGAP', context)
        block_number = self.parameterAsInt(parameters, 'BLOCKNUMBER', context)

        row_info = [int(x) for x in self.parameterAsString(parameters, 'ROWS', context).split(',')]
        col_info = [int(x) for x in self.parameterAsString(parameters, 'COLS', context).split(',')]
        plot_length = [float(x) for x in self.parameterAsString(parameters, 'PLOTLENGTH', context).split(',')]
        alley = [float(x) for x in self.parameterAsString(parameters, 'ALLEY', context).split(',')]

        ppc = self.parameterAsInt(parameters, 'PPC', context)
        one = self.parameterAsDouble(parameters, 'DIM1', context)
        two = self.parameterAsDouble(parameters, 'DIM2', context)
        three = self.parameterAsDouble(parameters, 'DIM3', context)
        four = self.parameterAsDouble(parameters, 'DIM4', context)

        # check for single values in CSV entries
        col_info = col_info if len(col_info) > 1 else col_info * block_number
        row_info = row_info if len(row_info) > 1 else row_info * block_number
        plot_length = plot_length if len(plot_length) > 1 else plot_length * block_number
        alley = alley if len(alley) > 1 else alley * block_number

        feedback.pushInfo(f'{len(col_info)=}, {len(row_info)=}, {len(plot_length)=}, {len(alley)=}')

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

        if len(margin) == 1:
            margin = (margin[0], margin[0])

        # calculate bearing
        dx = dest_x - origin_x
        dy = dest_y - origin_y
        bearing = math.degrees(math.atan2(dx, dy)) % 360
        bearing_radians = math.radians(bearing-180)

        feedback.pushInfo(f'Calculated bearing as: {bearing}')

        params = [(origin_x, origin_y), margin, bearing_radians, left, block_gap, row_info, col_info, plot_length, alley, ppc, one, two, three, four]
        output = _create_field_plan(*params)
        polygons, block_ids, row_ids, column_ids, plot_ids, guidance_lines_coords, guidance_lines_col, alley_lines_coords, alley_lines_row, alley_lines_block, boundary_lines_coords, boundary_lines_block = output

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

        if return_guidance_lines:
            crs = context.project().crs().authid()
            guidance_layer = QgsVectorLayer(f"Linestring?crs={crs}", GUIDANCE_LINE_LAYER_NAME, "memory")
            guidance_layer.startEditing()

            guidenace_provider = guidance_layer.dataProvider()
            guidance_fields = QgsFields()
            guidance_fields.append(QgsField("col", QMetaType.Int))
            guidenace_provider.addAttributes(guidance_fields)
            guidance_layer.updateFields()

            for (x1, y1, x2, y2), col in zip(guidance_lines_coords, guidance_lines_col):
                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(x1, y1), QgsPointXY(x2, y2)]))
                feat.setAttributes([col])
                guidenace_provider.addFeature(feat)
            
            guidance_layer.commitChanges()
            guidance_layer.updateExtents()
            context.temporaryLayerStore().addMapLayer(guidance_layer)
            context.addLayerToLoadOnCompletion(
                guidance_layer.id(),
                QgsProcessingContext.LayerDetails(
                    GUIDANCE_LINE_LAYER_NAME,
                    context.project(),
                    'FIELDPLANFIELDGUIDANCELINES'
                )
            )
            ret['FIELDPLANGUIDANCELINES'] = guidance_layer

        if return_alley_lines:
            crs = context.project().crs().authid()
            alley_layer = QgsVectorLayer(f"Linestring?crs={crs}", ALLEY_LINE_LAYER_NAME, "memory")
            alley_layer.startEditing()

            alley_provider = alley_layer.dataProvider()
            alley_fields = QgsFields()
            alley_fields.append(QgsField("row", QMetaType.Int))
            alley_fields.append(QgsField("block", QMetaType.Int))
            alley_provider.addAttributes(alley_fields)
            alley_layer.updateFields()

            for (x1, y1, x2, y2), row, block in zip(alley_lines_coords, alley_lines_row, alley_lines_block):
                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(x1, y1), QgsPointXY(x2, y2)]))
                feat.setAttributes([row, block])
                alley_provider.addFeature(feat)
            
            alley_layer.commitChanges()
            alley_layer.updateExtents()
            context.temporaryLayerStore().addMapLayer(alley_layer)
            context.addLayerToLoadOnCompletion(
                alley_layer.id(),
                QgsProcessingContext.LayerDetails(
                    ALLEY_LINE_LAYER_NAME,
                    context.project(),
                    'FIELDPLANALLEYLINES'
                )
            )
            ret['FIELDPLANALLEYLINES'] = alley_layer

        if return_block_boundaries:
            crs = context.project().crs().authid()
            boundaries_layer = QgsVectorLayer(f"Linestring?crs={crs}", BOUNDARY_LAYER_NAME, "memory")
            boundaries_layer.startEditing()

            boundaries_provider = boundaries_layer.dataProvider()
            boundaries_fields = QgsFields()
            boundaries_fields.append(QgsField("block", QMetaType.Int))
            boundaries_provider.addAttributes(boundaries_fields)
            boundaries_layer.updateFields()

            for (x1, y1, x2, y2), block in zip(boundary_lines_coords, boundary_lines_block):
                feat = QgsFeature()
                feat.setGeometry(QgsGeometry.fromPolylineXY([QgsPointXY(x1, y1), QgsPointXY(x2, y2)]))
                feat.setAttributes([block])
                boundaries_provider.addFeature(feat)
            
            boundaries_layer.commitChanges()
            boundaries_layer.updateExtents()
            context.temporaryLayerStore().addMapLayer(boundaries_layer)
            context.addLayerToLoadOnCompletion(
                boundaries_layer.id(),
                QgsProcessingContext.LayerDetails(
                    BOUNDARY_LAYER_NAME,
                    context.project(),
                    'FIELDPLANBLOCKBOUNDARIES'
                )
            )
            ret['FIELDPLANBLOCKBOUNDARIES'] = boundaries_layer

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
