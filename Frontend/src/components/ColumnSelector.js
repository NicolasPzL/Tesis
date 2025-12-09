// src/components/ColumnSelector.js
import React from 'react';
import { Form } from 'react-bootstrap';

/**
 * Componente para mostrar una lista de columnas con checkboxes.
 * @param {object} props
 * @param {string[]} props.columns - Array de nombres de columnas disponibles.
 * @param {string[]} props.selectedColumns - Array de nombres de columnas actualmente seleccionadas.
 * @param {function} props.onChange - Función callback llamada cuando la selección cambia, recibe el nuevo array de seleccionadas.
 */
const ColumnSelector = ({ columns, selectedColumns, onChange }) => {

  const handleCheckboxChange = (event) => {
    const { value, checked } = event.target;
    let newSelectedColumns;
    if (checked) {
      // Añadir columna a seleccionadas
      newSelectedColumns = [...selectedColumns, value];
    } else {
      // Quitar columna de seleccionadas
      newSelectedColumns = selectedColumns.filter(col => col !== value);
    }
    onChange(newSelectedColumns); // Notificar al padre del cambio
  };

  // Mensaje si no hay columnas (archivo vacío o error previo)
  if (!columns || columns.length === 0) {
      return <p className="text-muted fst-italic">No columns available to select. Check the uploaded file.</p>;
  }

  return (
    <div className="column-selector">
      <h6>Select columns to analyze:</h6>
      <Form>
        {/* Crear un checkbox por cada columna disponible */}
        {columns.map(column => (
          <Form.Check
            type="checkbox"
            id={`col-checkbox-${column}`} // ID único para el label
            key={column}
            label={column} // Nombre de la columna como etiqueta
            value={column} // Valor del checkbox es el nombre de la columna
            checked={selectedColumns.includes(column)} // Marcar si está en el array de seleccionadas
            onChange={handleCheckboxChange} // Manejador de cambio
            className="mb-2" // Margen inferior
          />
        ))}
      </Form>
      <p className="mt-2 text-muted small">
        {selectedColumns.length} / {columns.length} column(s) selected.
      </p>
    </div>
  );
};

export default ColumnSelector;