from typing import List, Dict, Any, Optional
import openpyxl
from dataclasses import dataclass
from openpyxl.utils import get_column_letter
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CellData:
    """Represents a cell with its metadata."""
    sheet_name: str
    row: int
    column: int
    value: Any
    formula: Optional[str] = None
    address: str = ""
    data_type: str = ""

@dataclass
class SemanticChunk:
    """Represents a semantic chunk of spreadsheet data."""
    content: str
    metadata: Dict[str, Any]
    chunk_type: str  # 'cell', 'row', 'column', 'sheet_summary', 'formula'

class SpreadsheetReader:
    """Reads and parses Excel files to extract semantic information."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.workbook = None
        self.cell_data: List[CellData] = []
    
    def _sanitize_metadata_for_chroma(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert metadata to ChromaDB-compatible format.
        ChromaDB only accepts: str, int, float, bool, or None
        """
        sanitized = {}
        
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = None
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                # Convert list to comma-separated string
                sanitized[key] = ", ".join(str(item) for item in value)
            elif isinstance(value, dict):
                # Convert dict to JSON string
                sanitized[key] = json.dumps(value)
            else:
                # Convert anything else to string
                sanitized[key] = str(value)
        
        return sanitized
    
    def _extract_formula_safely(self, cell) -> Optional[str]:
        """
        Extract formula from a cell using every possible method.
        This method will NEVER throw an AttributeError!
        """
        try:
            # Method 1: Check if cell has data_type = 'f' (formula)
            if hasattr(cell, 'data_type') and cell.data_type == 'f':
                for attr_name in ['formula', '_formula', 'f']:
                    try:
                        formula_val = getattr(cell, attr_name, None)
                        if formula_val and str(formula_val).strip() and str(formula_val) != '=':
                            return str(formula_val)
                    except (AttributeError, TypeError):
                        continue
            
            # Method 2: Try direct attribute access with getattr (safe)
            try:
                formula_val = getattr(cell, 'formula', None)
                if formula_val and str(formula_val).strip() and str(formula_val) != '=':
                    return str(formula_val)
            except (AttributeError, TypeError):
                pass
            
            # Method 3: Check if the cell value starts with '=' (indicates formula)
            if isinstance(cell.value, str) and cell.value.startswith('='):
                return cell.value
            
            return None
        
        except Exception as e:
            logger.warning(f"Formula extraction failed for cell {getattr(cell, 'coordinate', 'unknown')}: {e}")
            return None
        
    def load_workbook(self) -> None:
        """Load the Excel workbook with formula preservation."""
        try:
            # Load with data_only=False to preserve formulas and read formulas
            # keep_links=False to avoid issues with external links
            self.workbook = openpyxl.load_workbook(
                self.file_path, 
                data_only=False, 
                keep_vba=False,
                read_only=False,
                keep_links=False
            )
            logger.info(f"Successfully loaded workbook with {len(self.workbook.sheetnames)} sheets")
        except Exception as e:
            raise ValueError(f"Failed to load workbook: {str(e)}")
    
    def extract_cell_data(self) -> List[CellData]:
        """Extract all cell data from the workbook."""
        if not self.workbook:
            self.load_workbook()
        
        self.cell_data = []
        
        for sheet_name in self.workbook.sheetnames:
            sheet = self.workbook[sheet_name]
            
            for row in sheet.iter_rows():
                for cell in row:
                    if cell.value is not None:
                        try:
                            # Handle formula attribute safely using helper method
                            formula = self._extract_formula_safely(cell)
                            
                            # Debug logging for formula detection
                            if formula:
                                logger.debug(f"Found formula in {cell.coordinate}: {formula}")
                            
                            # Handle data_type attribute safely
                            data_type = ''
                            if hasattr(cell, 'data_type'):
                                data_type = str(cell.data_type)
                            else:
                                # Infer data type from value
                                if isinstance(cell.value, (int, float)):
                                    data_type = 'n'  # numeric
                                elif isinstance(cell.value, str):
                                    data_type = 's'  # string
                                elif isinstance(cell.value, bool):
                                    data_type = 'b'  # boolean
                                else:
                                    data_type = 's'  # default to string
                            
                            cell_info = CellData(
                                sheet_name=sheet_name,
                                row=cell.row,
                                column=cell.column,
                                value=cell.value,
                                formula=formula,
                                address=cell.coordinate,
                                data_type=data_type
                            )
                            self.cell_data.append(cell_info)
                            
                        except Exception as e:
                            # Log the error but continue processing
                            logger.warning(f"Error processing cell {cell.coordinate} in sheet '{sheet_name}': {e}")
                            # Still add basic cell info even if some attributes fail
                            cell_info = CellData(
                                sheet_name=sheet_name,
                                row=cell.row,
                                column=cell.column,
                                value=cell.value,
                                formula=None,
                                address=cell.coordinate,
                                data_type='s'  # default
                            )
                            self.cell_data.append(cell_info)
        
        return self.cell_data
    
    def create_semantic_chunks(self) -> List[SemanticChunk]:
        """Create semantic chunks from the extracted cell data."""
        chunks = []
        
        # Group cells by sheet
        sheets_data = {}
        for cell in self.cell_data:
            if cell.sheet_name not in sheets_data:
                sheets_data[cell.sheet_name] = []
            sheets_data[cell.sheet_name].append(cell)
        
        for sheet_name, cells in sheets_data.items():
            # Create sheet summary chunk
            sheet_summary = self._create_sheet_summary(sheet_name, cells)
            chunks.append(sheet_summary)
            
            # Create individual cell chunks
            for cell in cells:
                cell_chunk = self._create_cell_chunk(cell)
                chunks.append(cell_chunk)
                
                # Create formula chunks if applicable
                if cell.formula and cell.formula.strip():
                    formula_chunk = self._create_formula_chunk(cell)
                    chunks.append(formula_chunk)
            
            # Create row and column chunks
            row_chunks = self._create_row_chunks(sheet_name, cells)
            column_chunks = self._create_column_chunks(sheet_name, cells)
            chunks.extend(row_chunks)
            chunks.extend(column_chunks)
        
        return chunks
    
    def _create_sheet_summary(self, sheet_name: str, cells: List[CellData]) -> SemanticChunk:
        """Create a summary chunk for a sheet."""
        non_empty_cells = [c for c in cells if c.value is not None]
        formulas = [c for c in cells if c.formula and c.formula.strip()]
        
        content = f"Sheet '{sheet_name}' contains {len(non_empty_cells)} non-empty cells."
        if formulas:
            content += f" It has {len(formulas)} formulas."
        
        # Extract headers (first row)
        headers = [c.value for c in cells if c.row == 1 and c.value is not None]
        if headers:
            content += f" Column headers: {', '.join(str(h) for h in headers)}."
        
        metadata = self._sanitize_metadata_for_chroma({
            "sheet_name": sheet_name,
            "total_cells": len(cells),
            "non_empty_cells": len(non_empty_cells),
            "formulas_count": len(formulas),
            "headers": headers
        })
        
        return SemanticChunk(
            content=content,
            metadata=metadata,
            chunk_type="sheet_summary"
        )
    
    def _create_cell_chunk(self, cell: CellData) -> SemanticChunk:
        """Create a semantic chunk for a single cell."""
        content = f"In sheet '{cell.sheet_name}', cell {cell.address} contains: {cell.value}"
        
        metadata = self._sanitize_metadata_for_chroma({
            "sheet_name": cell.sheet_name,
            "address": cell.address,
            "row": cell.row,
            "column": cell.column,
            "value": str(cell.value),
            "data_type": cell.data_type
        })
        
        return SemanticChunk(
            content=content,
            metadata=metadata,
            chunk_type="cell"
        )
    
    def _create_formula_chunk(self, cell: CellData) -> SemanticChunk:
        """Create a semantic chunk for a formula."""
        # Safety check for formula
        formula_text = cell.formula if cell.formula else "Unknown formula"
        content = f"Formula in sheet '{cell.sheet_name}', cell {cell.address}: {formula_text}"
        
        metadata = self._sanitize_metadata_for_chroma({
            "sheet_name": cell.sheet_name,
            "address": cell.address,
            "formula": formula_text,
            "result": str(cell.value)
        })
        
        return SemanticChunk(
            content=content,
            metadata=metadata,
            chunk_type="formula"
        )
    
    def _create_row_chunks(self, sheet_name: str, cells: List[CellData]) -> List[SemanticChunk]:
        """Create semantic chunks for rows."""
        chunks = []
        
        # Group cells by row
        rows = {}
        for cell in cells:
            if cell.row not in rows:
                rows[cell.row] = []
            rows[cell.row].append(cell)
        
        for row_num, row_cells in rows.items():
            values = [str(c.value) for c in row_cells if c.value is not None]
            if len(values) > 1:  # Only create row chunks for multi-cell rows
                content = f"Row {row_num} in sheet '{sheet_name}' contains: {', '.join(values)}"
                
                metadata = self._sanitize_metadata_for_chroma({
                    "sheet_name": sheet_name,
                    "row": row_num,
                    "cells_count": len(row_cells),
                    "values": values
                })
                
                chunks.append(SemanticChunk(
                    content=content,
                    metadata=metadata,
                    chunk_type="row"
                ))
        
        return chunks
    
    def _create_column_chunks(self, sheet_name: str, cells: List[CellData]) -> List[SemanticChunk]:
        """Create semantic chunks for columns."""
        chunks = []
        
        # Group cells by column
        columns = {}
        for cell in cells:
            col_letter = get_column_letter(cell.column)
            if col_letter not in columns:
                columns[col_letter] = []
            columns[col_letter].append(cell)
        
        for col_letter, col_cells in columns.items():
            values = [str(c.value) for c in col_cells if c.value is not None]
            if len(values) > 1:  # Only create column chunks for multi-cell columns
                content = f"Column {col_letter} in sheet '{sheet_name}' contains: {', '.join(values)}"
                
                metadata = self._sanitize_metadata_for_chroma({
                    "sheet_name": sheet_name,
                    "column": col_letter,
                    "cells_count": len(col_cells),
                    "values": values
                })
                
                chunks.append(SemanticChunk(
                    content=content,
                    metadata=metadata,
                    chunk_type="column"
                ))
        
        return chunks
    
    def process_file(self) -> List[SemanticChunk]:
        """Process the entire spreadsheet file and return semantic chunks."""
        self.extract_cell_data()
        return self.create_semantic_chunks()