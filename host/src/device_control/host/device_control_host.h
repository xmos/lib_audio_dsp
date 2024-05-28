// Copyright 2024 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef __control_host_h__
#define __control_host_h__

#include "device_control_shared.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \defgroup device_control_host
 *
 * The public API for using the device control host library
 * @{
 */

#if USE_XSCOPE  || __DOXYGEN__
/** Initialize the xscope host interface
 *
 *  \param host_str    String containing the name of the xscope host. Eg. "localhost"
 *  \param port_str    String containing the port number of the xscope host
 *
 *  \returns           Whether the initialization was successful or not
 */
control_ret_t control_init_xscope(const char *host_str, const char *port_str);
/** Shutdown the xscope host interface
 *
 *  \returns           Whether the shutdown was successful or not
 */
control_ret_t control_cleanup_xscope(void);
#endif

#if USE_I2C || __DOXYGEN__
/** Initialize the I2C host (master) interface
 *
 *  \param i2c_slave_address    I2C address of the slave (controlled device)
 *
 *  \returns                    Whether the initialization was successful or not
 */
control_ret_t control_init_i2c(unsigned char i2c_slave_address);
/** Shutdown the I2C host (master) interface connection
 *
 *  \returns           Whether the shutdown was successful or not
 */
control_ret_t control_cleanup_i2c(void);
#endif

#if USE_USB || __DOXYGEN__
/** Initialize the USB host interface
 *
 *  \param vendor_id     Vendor ID of controlled USB device
 *  \param product_id    Product ID of controlled USB device
 *  \param interface_num USB Control interface number of controlled device
 *
 *  \returns           Whether the initialization was successful or not
 */
control_ret_t control_init_usb(int vendor_id, int product_id, int interface_num);
/** Shutdown the USB host interface connection
 *
 *  \returns           Whether the shutdown was successful or not
 */
control_ret_t control_cleanup_usb(void);
#endif

#if (!USE_USB && !USE_XSCOPE)
#error "Please specify transport for device control using USE_xxx define in build file"
#error "Eg. -DUSE_USB=1 or -DUSE_XSCOPE=1"
#endif

/** Checks to see that the version of control library in the device is the same as the host
 *
 *  \param version      Reference to control version variable that is set on this call
 *
 *  \returns            Whether the checking of control library version was successful or not
 */
control_ret_t control_query_version(control_version_t *version);

/** Request to write to controllable resource inside the device. The command consists of a resource ID,
 *  command and a byte payload of length payload_len.
 *
 *  \param resid        Resource ID. Indicates which resource the command is intended for
 *  \param cmd          Command code. Note that this will be in the range 0x80 to 0xFF
 *                      because bit 7 set indiciates a write command
 *  \param payload      Array of bytes which constitutes the data payload
 *  \param payload_len  Size of the payload in bytes
 *
 *  \returns            Whether the write to the device was successful or not
 */
control_ret_t
control_write_command(control_resid_t resid, control_cmd_t cmd,
                      const uint8_t payload[], size_t payload_len);

/** Request to read from controllable resource inside the device. The command consists of a resource ID,
 *  command and a byte payload of length payload_len.
 *
 *  \param resid        Resource ID. Indicates which resource the command is intended for
 *  \param cmd          Command code. Note that this will be in the range 0x80 to 0xFF
 *                      because bit 7 set indiciates a write command
 *  \param payload      Array of bytes which constitutes the data payload
 *  \param payload_len  Size of the payload in bytes
 *
 *  \returns            Whether the read from the device was successful or not
 */
control_ret_t
control_read_command(control_resid_t resid, control_cmd_t cmd,
                     uint8_t payload[], size_t payload_len);

#ifdef __cplusplus
}
#endif

/**@}*/

#endif // __control_host_h__
